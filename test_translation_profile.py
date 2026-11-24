import json,unittest
from unittest.mock import patch
import numpy as np
import engine
from translation_profile import load_profile,relevant_terms
from typesetting import fit_adaptive,fit_text

class TranslationProfileTests(unittest.TestCase):
    def test_colloquial_negation_cannot_become_a_positive_request(self):
        from text_rules import invalid_translation,source_grammar_hint
        self.assertTrue(invalid_translation('びっくりさせんなよ。','吓唬我一下。'))
        self.assertFalse(invalid_translation('びっくりさせんなよ。','别吓我。'))
        self.assertTrue(invalid_translation('そんなこと言ってねえよ。','别这么说。'))
        self.assertFalse(invalid_translation('そんなこと言ってねえよ。','我可没说过这种话。'))
        for source in ('ねえ、聞いて。','言ってね。','食べてねえか？'):
            self.assertEqual(source_grammar_hint(source),'')

    def test_contraction_expansion_leaves_requests_and_questions_intact(self):
        from text_rules import translation_input
        self.assertEqual(translation_input('食べてねえよ。'),'食べていないよ。')
        self.assertEqual(translation_input('無理させんなよ。'),'無理させるなよ。')
        self.assertEqual(translation_input('邪魔すんな。'),'邪魔するな。')
        for text in ('言ってね。','ねえ、聞いて。','食べてねえか？','Do not do that.'):
            self.assertEqual(translation_input(text),text)
    def test_whole_word_matching_and_inflections(self):
        profile=load_profile()
        self.assertNotIn('TRAP',relevant_terms(profile,'The strap broke.'))
        self.assertIn('TRAP',relevant_terms(profile,'A TRAP.'))
        self.assertIn('CREAMPIE',relevant_terms(profile,'creampied'))
        self.assertIn('エッチ',relevant_terms(profile,'エッチな話'))

    def test_initial_and_retry_share_custom_profile_and_user_priority(self):
        translator=engine.LocalTranslator(lambda _:None);translator.process=True;translator.url='http://127.0.0.1'
        translator.profile={'instructions':['测试用的统一语气配置'], 'terms':[{'term':'hello','guidance':'测试词义提示'}]}
        calls=[]
        class Response:
            def raise_for_status(self):pass
            def json(self):
                content=json.dumps({'items':[{'id':0,'kind':'dialogue','zh':'你好。'}]}) if len(calls)==1 else '你好。'
                return {'choices':[{'message':{'content':content}}]}
        def post(url,**kwargs):calls.append(kwargs['json']);return Response()
        r={'id':3,'source':'HELLO','kind':'dialogue'}
        with patch.object(translator.http,'post',post):
            translator.translate([r],'en','用户优先词表',[])
            self.assertTrue(translator.retry_translation(r,'en','用户优先词表'))
        for body in calls:
            self.assertIn('测试用的统一语气配置',body['messages'][0]['content'])
            self.assertIn('测试词义提示',body['messages'][1]['content'])
            self.assertIn('用户优先词表',body['messages'][1]['content'])
        translator.http.close()

    def test_sparse_balloon_uses_room_but_stays_inside_contour(self):
        bubble={'box':[0,0,300,300],'center':(150,150),'safe':np.ones((300,300),np.uint8)}
        box=[115,115,185,185];text='明天一起回家吧'
        fitted=fit_adaptive(text,box,False,engine.FONT,bubble,text)
        self.assertGreater(fitted['occupancy'],.10)
        self.assertLess(fitted['occupancy'],.43)
        x,y,X,Y=fitted['bounds']
        self.assertTrue(bubble['safe'][y:Y,x:X][np.asarray(fitted['tile'])>0].all())

    def test_dense_balloon_and_floating_text_are_not_enlarged(self):
        bubble={'box':[0,0,300,300],'center':(150,150),'safe':np.ones((300,300),np.uint8)}
        box=[45,45,255,255];text='我们明天在门口见，到时候再一起过去。'
        old=fit_text(text,box,False,engine.FONT,bubble,text)
        new=fit_adaptive(text,box,False,engine.FONT,bubble,text)
        self.assertEqual(old['size'],new['size'])
        loose=fit_text(text,box,False,engine.FONT,None,text)
        same=fit_adaptive(text,box,False,engine.FONT,None,text)
        self.assertEqual(loose['size'],same['size'])

if __name__=='__main__':unittest.main()
