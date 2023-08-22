from train import one_sentence_translate

sentence="こんにちは。\n今日はいい天気ですね。\nどこかに出かけませんか？"
one_sentence_translate(sentence, beam_search=True)
