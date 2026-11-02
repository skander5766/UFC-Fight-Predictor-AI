"""UI integration checks through Streamlit AppTest; no inference reimplementation."""
from datetime import date
import hashlib
from pathlib import Path
import runpy
import unittest

from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'app/streamlit_app.py'


class StreamlitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths=[p for folder in (ROOT/'data',ROOT/'models',ROOT/'src/prediction') for p in folder.rglob('*') if p.is_file()]
        cls.hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.paths}

    @classmethod
    def tearDownClass(cls):
        assert cls.hashes=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.paths}

    def test_select_predict_swap_and_stale_result(self):
        app=AppTest.from_file(str(APP),default_timeout=40).run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.selectbox(key='fighter_a').options),0)
        self.assertTrue(app.selectbox(key='fighter_a').disabled)
        self.assertTrue(app.selectbox(key='fighter_b').disabled)
        app.selectbox(key='weight_class').select('Heavyweight').run()
        self.assertGreater(len(app.selectbox(key='fighter_a').options),100)
        self.assertTrue(app.button(key='predict').disabled)
        app.selectbox(key='fighter_a').select('Brock Lesnar').run()
        self.assertNotIn('Brock Lesnar',app.selectbox(key='fighter_b').options)
        app.selectbox(key='fighter_b').select('Mark Hunt').run()
        app.date_input(key='prediction_date').set_value(date(2016,7,9)).run()
        app.button(key='predict').click().run()
        self.assertEqual(len(app.exception),0)
        first=app.session_state['prediction_result']
        expected=runpy.run_path(str(ROOT/'src/prediction/predict_fight_v2.py'))['predict_fight_v2']('Brock Lesnar','Mark Hunt','2016-07-09',3)
        self.assertEqual(first,expected)
        self.assertEqual([x.value for x in app.subheader][:4],['WIN PROBABILITY','METHOD GIVEN WIN','OVERALL WIN-BY-METHOD','FINISH ROUND PRIOR'])
        self.assertEqual(len(app.table),3)
        # All charts display the engine outputs on the same fixed scale.
        charts=[x.value for x in app.markdown if 'class="prob-chart"' in x.value]
        self.assertEqual(len(charts),3)
        for chart,key in zip(charts[:2],['method_given_win','joint_winner_method']):
            for side in ('A','B'):
                for value in first[key][side].values():
                    self.assertIn(f'width:{value * 100:.8f}%',chart)
                    self.assertIn(f'{value:.1%}',chart)
            self.assertIn('Scale: 0% to 100%',chart)
            self.assertNotIn('<span>Probability</span>',chart)
        for i in range(1,4):
            self.assertIn(f'Round {i}',charts[2])
        for i in (4,5):
            self.assertNotIn(f'Round {i}',charts[2])
        self.assertEqual(list(app.table[2].value['Round']),['R1','R2','R3'])
        self.assertEqual(list(app.table[2].value['Historical probability']),
                         [f"{first['round_given_finish']['probabilities'][f'ROUND_{i}']:.2%}" for i in range(1,4)])
        details=next(e for e in app.expander if e.label=='Scientific details & matchup warnings')
        self.assertFalse(details.proto.expanded)
        self.assertEqual([w.value for w in details.warning],first['warnings'])
        self.assertTrue(any('Probabilities are uncertain estimates for supported outcomes' in i.value
                            and 'Draws, no contests, DQs and overturned outcomes' in i.value for i in app.info))
        self.assertEqual([w.value for w in app.warning],first['warnings'])
        self.assertTrue(any('Not ML predictions' in c.value for c in app.caption))
        self.assertEqual(first['round_given_finish']['probabilities']['ROUND_4'],0)
        app.selectbox(key='scheduled_rounds').select(5).run()
        self.assertNotIn('prediction_result',app.session_state)
        app.button(key='predict').click().run()
        self.assertGreater(app.session_state['prediction_result']['round_given_finish']['probabilities']['ROUND_4'],0)
        round_chart=[x.value for x in app.markdown if 'class="prob-chart"' in x.value][-1]
        for i in range(1,6):
            self.assertIn(f'Round {i}',round_chart)
        self.assertEqual(list(app.table[2].value['Round']),['R1','R2','R3','R4','R5'])
        five=app.session_state['prediction_result']
        self.assertEqual(list(app.table[2].value['Historical probability']),
                         [f"{five['round_given_finish']['probabilities'][f'ROUND_{i}']:.2%}" for i in range(1,6)])
        self.assertTrue(any('1653' in w.value for w in app.warning))
        app.button(key='swap').click().run()
        self.assertEqual(app.selectbox(key='fighter_a').value,'Mark Hunt')
        self.assertEqual(app.selectbox(key='fighter_b').value,'Brock Lesnar')
        self.assertNotIn('prediction_result',app.session_state)
        app.button(key='predict').click().run()
        self.assertAlmostEqual(first['winner_probabilities']['A'],1-app.session_state['prediction_result']['winner_probabilities']['A'],places=12)
        app.date_input(key='prediction_date').set_value(date(2016,7,10)).run()
        self.assertNotIn('prediction_result',app.session_state)
        self.assertEqual(len(app.exception),0)

    def test_chart_escapes_names_and_preserves_zero_and_full_probability(self):
        module=runpy.run_path(str(APP))
        chart=module['probability_chart'](
            [('KO < TKO',{'A':0.,'B':1.})],
            {'A':'A <script>alert(1)</script>','B':'B "opponent"'},'A & B',
        )
        self.assertNotIn('<script>',chart)
        self.assertIn('&lt;script&gt;',chart)
        self.assertIn('KO &lt; TKO',chart)
        self.assertIn('role="img"',chart)
        self.assertIn('aria-label="A &amp; B:',chart)
        self.assertIn('width:0.00000000%',chart)
        self.assertIn('width:100.00000000%',chart)
        self.assertIn('0.0%',chart)
        self.assertIn('100.0%',chart)

    def test_date_without_division_history_clears_selection_and_result(self):
        app=AppTest.from_file(str(APP),default_timeout=40).run()
        app.selectbox(key='weight_class').select('Heavyweight').run()
        app.selectbox(key='fighter_a').select('Brock Lesnar').run()
        app.selectbox(key='fighter_b').select('Mark Hunt').run()
        app.button(key='predict').click().run()
        self.assertIn('prediction_result',app.session_state)
        app.date_input(key='prediction_date').set_value(date(1900,1,1)).run()
        self.assertEqual(len(app.exception),0)
        self.assertIsNone(app.selectbox(key='weight_class').value)
        self.assertIsNone(app.selectbox(key='fighter_a').value)
        self.assertIsNone(app.selectbox(key='fighter_b').value)
        self.assertTrue(app.selectbox(key='fighter_a').disabled)
        self.assertTrue(app.button(key='predict').disabled)
        self.assertNotIn('prediction_result',app.session_state)
        self.assertEqual(len(app.table),0)

    def test_underlying_path_input_errors_and_limited_history(self):
        module=runpy.run_path(str(APP))
        signature=module['snapshot_signature']()
        for a,b,day in [('Brock Lesnar','Brock Lesnar','2016-07-09'),
                        ('Not A Real Fighter','Mark Hunt','2016-07-09'),
                        ('Brock Lesnar','Mark Hunt','not-a-date')]:
            with self.subTest(a=a,day=day),self.assertRaises(ValueError):
                module['run_prediction'](a,b,day,signature,3)
        result=module['run_prediction']('Brock Lesnar','Frank Mir','2008-02-02',signature,3)
        self.assertTrue(any('0 prior fights' in w for w in result['warnings']))

    def test_missing_artifact_friendly_error(self):
        source=f"""import runpy
module=runpy.run_path({str(APP)!r})
def unavailable():
    raise FileNotFoundError('missing artifact')
module['main'].__globals__['snapshot_signature']=unavailable
module['main']()
"""
        app=AppTest.from_string(source,default_timeout=40).run()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any('files are missing' in error.value for error in app.error))

    def test_prediction_failure_friendly_error(self):
        source=f"""import runpy
module=runpy.run_path({str(APP)!r})
def fail(*args):
    raise RuntimeError('internal failure')
module['main'].__globals__['run_prediction']=fail
module['main']()
"""
        app=AppTest.from_string(source,default_timeout=40).run()
        app.selectbox(key='weight_class').select('Heavyweight').run()
        app.selectbox(key='fighter_a').select('Brock Lesnar').run()
        app.selectbox(key='fighter_b').select('Mark Hunt').run()
        app.button(key='predict').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any('could not be produced' in error.value for error in app.error))
        self.assertNotIn('prediction_result',app.session_state)

    def test_invalid_artifact_friendly_error(self):
        source=f"""import runpy
module=runpy.run_path({str(APP)!r})
def invalid(*args):
    raise ValueError('Artifact checksum mismatch.')
module['main'].__globals__['load_engine']=invalid
module['main']()
"""
        app=AppTest.from_string(source,default_timeout=40).run()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any('could not be loaded or validated' in error.value for error in app.error))
        self.assertEqual(len(app.table),0)
        self.assertNotIn('prediction_result',app.session_state)


if __name__=='__main__':unittest.main()
