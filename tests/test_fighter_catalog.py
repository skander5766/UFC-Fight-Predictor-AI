from pathlib import Path
import runpy
import unittest
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
M=runpy.run_path(str(ROOT/'src/data/fighter_catalog.py'))

class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.data=pd.DataFrame({'fighter_1':['Alpha','Alpha','Zeta'], 'fighter_2':['Beta','Gamma','Beta'],
                               'WEIGHTCLASS':['UFC Lightweight Title Bout','Welterweight Bout','Lightweight Bout'],
                               'event_date':['2010-01-01','2015-01-01',None]})
        self.original=self.data.copy(deep=True)
        self.catalog=M['FighterCatalog'](self.data)

    def test_extraction_and_normalization(self):
        self.assertEqual(self.catalog.raw_counts.sum(),3)
        cases={'UFC Interim Lightweight Title Bout':'Lightweight',
               "Road to UFC 3 Women's Strawweight Tournament TitleBout":"Women's Strawweight",
               'Ultimate Fighter 1 Light Heavyweight Tournament Title Bout':'Light Heavyweight',
               'Super Heavyweight Bout':'Super Heavyweight','Catch Weight Bout':'Catch Weight',
               'UFC Superfight Championship Bout':M['UNSPECIFIED'],'UFC 2 Tournament Title Bout':M['UNSPECIFIED']}
        for raw,expected in cases.items():self.assertEqual(M['normalize_division'](raw),expected)

    def test_membership_sorting_and_sharing(self):
        self.assertEqual(self.catalog.get_fighter_weight_classes('Beta'),['Lightweight'])
        self.assertEqual(self.catalog.get_fighter_weight_classes(' alpha '),['Lightweight','Welterweight'])
        self.assertEqual(self.catalog.get_fighters_for_weight_class('Lightweight'),['Alpha','Beta','Zeta'])
        self.assertTrue(self.catalog.fighters_share_weight_class('Alpha','Gamma'))
        self.assertFalse(self.catalog.fighters_share_weight_class('Beta','Gamma'))
        self.assertFalse(self.catalog.fighters_share_weight_class('Alpha','ALPHA'))

    def test_strict_date_and_undated(self):
        self.assertEqual(self.catalog.get_fighter_weight_classes('Alpha','2010-01-01'),[])
        self.assertEqual(self.catalog.get_fighter_weight_classes('Alpha','2015-01-01'),['Lightweight'])
        self.assertEqual(self.catalog.get_fighters_for_weight_class('Lightweight','2016-01-01'),['Alpha','Beta'])

    def test_status_unknown_not_inferred(self):
        self.assertIsNone(self.catalog.is_retired('Alpha','2099-01-01'))
        self.assertEqual(self.catalog.get_active_fighters_for_weight_class('Lightweight'),['Alpha','Beta','Zeta'])
        self.assertEqual(self.catalog.get_active_fighters_for_weight_class('Lightweight',include_unknown=False),[])
        pd.testing.assert_frame_equal(self.data,self.original)

    def test_verified_status_only_filters_application(self):
        evidence=[{'fighter':'Alpha','status':'retired','verified':True,'source':'test evidence only',
                   'effective_date':'2016-01-01','verified_date':'2016-01-02'}]
        catalog=M['FighterCatalog'](self.data,evidence)
        self.assertIsNone(catalog.is_retired('Alpha','2016-01-01'))
        self.assertTrue(catalog.is_retired('Alpha','2017-01-01'))
        self.assertEqual(catalog.get_active_fighters_for_weight_class('Lightweight','2017-01-01'),['Beta'])
        self.assertIn('Alpha',catalog.get_fighters_for_weight_class('Lightweight'))
        pd.testing.assert_frame_equal(self.data,self.original)
        evidence[0]['verified']=False
        with self.assertRaises(ValueError):M['FighterCatalog'](self.data,evidence)

    def test_ui_division_change_clears_fighters(self):
        app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=40).run()
        app.selectbox(key='weight_class').select('Heavyweight').run()
        app.selectbox(key='fighter_a').select('Brock Lesnar').run()
        self.assertNotIn('Brock Lesnar',app.selectbox(key='fighter_b').options)
        app.selectbox(key='weight_class').select("Women's Strawweight").run()
        self.assertIsNone(app.selectbox(key='fighter_a').value)
        self.assertNotIn('Brock Lesnar',app.selectbox(key='fighter_a').options)
        self.assertTrue(app.button(key='predict').disabled)
        self.assertEqual(len(app.exception),0)

if __name__=='__main__':unittest.main()
