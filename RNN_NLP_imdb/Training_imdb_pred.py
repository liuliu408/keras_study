#coding:utf-8
from keras.preprocessing.text import Tokenizer #建立字典
from keras.preprocessing import sequence #统一化维度
from keras.models import load_model
import os
from keras.models import Sequential



def pre_review(newtext):
    SentimentDict = {1: '负面的', 0: '正面的'}
    token = Tokenizer(num_words=500)
    newtext = [newtext]
    token.fit_on_texts(newtext)
    input_seq = token.texts_to_sequences(newtext)
    pad = sequence.pad_sequences(input_seq,maxlen=200)
    model = load_model('text.h5')
    predict = model.predict_classes(pad)
    print predict
#neg review
#pre_review('As a fan of the original Disney film (Personally I feel it’s their masterpiece) I was taken aback to the fact that a new version was in the making. Still excited I had high hopes for the film. Most of was shattered in the first 10 minutes. Campy acting with badly performed singing starts off a long journey holding hands with some of the worst CGI Hollywood have managed to but to screen in ages. A film that is over 50% GCI, should focus on making that part believable, unfortunately for this film, it’s far from that. It looks like the original film was ripped apart frame by frame and the beautiful hand-painted drawings have been replaced with digital caricatures. Besides CGI that is bad, it’s mostly creepy. As the little teacup boy will give me nightmares for several nights to come. Emma Watson plays the same character as she always does, with very little acting effort and very little conviction as Belle. Although I can see why she was cast in the film based on merits, she is far from the right choice for the role. Dan Stevens does alright under as some motion captured dead-eyed Beast, but his performance feels flat as well. Luke Evans makes for a great pompous Gaston, but a character that has little depth doesn’t really make for a great viewing experience. Josh Gad is a great comic relief just like the original movie’s LeFou. Other than that, none of the cast stands out enough for me to remember them. Human or CHI creature. I was just bored through out the whole experience. And for a project costing $160 000 000, I can see why the PR department is pushing it so hard because they really need to get some cash back on this pile of wet stinky CGI-fur!All and all, I might be bias from really loving Disney’s first adaptation. That for me marks the high-point of all their work, perfectly combining the skills of their animators along with some CGI in a majestic blend. This film however is more like the bucket you wash off your paintbrush in, it has all the same colors, but muddled with water and to thin to make a captivating story from. The film is quite frankly not worth your time, you would be better off watching the original one more time.')
#pos review
pre_review('BLACK WATER is a thriller that manages to completely transcend it’s limitations (it’s an indie flick) by continually subverting expectations to emerge as an intense experience.In the tradition of all good animal centered thrillers ie Jaws, The Edge, the original Cat People, the directors know that restraint and what isn’t shown are the best ways to pack a punch. The performances are real and gripping, the crocdodile is extremely well done, indeed if the Black Water website is to be believed that’s because they used real crocs and the swamp location is fabulous.If you are after a B-grade gore fest croc romp forget Black Water but if you want a clever, suspenseful ride that will have you fearing the water and wondering what the hell would I do if i was up that tree then it’s a must see.')
