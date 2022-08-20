from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import re
import spacy
from spacy.symbols import ORTH, NORM, LEMMA
from spacy.lang.char_classes import LIST_PUNCT, LIST_ELLIPSES, LIST_QUOTES, LIST_CURRENCY
from spacy.lang.char_classes import LIST_ICONS, HYPHENS, CURRENCY, UNITS
from spacy.lang.char_classes import CONCAT_QUOTES, ALPHA_LOWER, ALPHA_UPPER, ALPHA, PUNCT
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex


nlp2 = spacy.load('en_core_web_sm')
##### text preprocessor for ekphrasis
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'date', 'number'],
    # terms that will be annotated
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter",

    # corpus from which the word statistics are going to be used 
    # for spell correction
    # corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

#### AutoTokenization
def custom_tokenize(sent, tokenizer, max_length=512):
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    encoded_sent = tokenizer.encode(
        sent,  # Sentence to encode.
        add_special_tokens=False)
    return encoded_sent


# input: text
# process: ekphrasis preprocesser + some extra processing
# output: list of tokens
def ek_extra_preprocess(text, params, tokenizer):
    remove_words = ['<allcaps>', '</allcaps>', '<hashtag>', '</hashtag>', '<elongated>', '<emphasis>', '<repeated>',
                    '\'', 's']
    word_list = text_processor.pre_process_doc(text)
    if (params['include_special']):
        pass
    else:
        word_list = list(filter(lambda a: a not in remove_words, word_list))
    sent = " ".join(word_list)
    sent = re.sub(r"[<\*>]", " ", sent)
    sub_word_list = custom_tokenize(sent, tokenizer)
    return sub_word_list


# input: text
# process: remove html tags
# output: text with no html tags
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


##### Preprocessing queries for raw text not needed for implementation
special_cases = {}
# Times
for h in range(1, 12 + 1):
    for period in ["a.m.", "am"]:
        special_cases["%d%s" % (h, period)] = [
            {ORTH: "%d" % h},
            {ORTH: period, LEMMA: "a.m.", NORM: "a.m."},
        ]
    for period in ["p.m.", "pm"]:
        special_cases["%d%s" % (h, period)] = [
            {ORTH: "%d" % h},
            {ORTH: period, LEMMA: "p.m.", NORM: "p.m."},
        ]

for orth in [
    "a.m.",
    "Adm.",
    "Bros.",
    "co.",
    "Co.",
    "Corp.",
    "D.C.",
    "Dr.",
    "e.g.",
    "E.g.",
    "E.G.",
    "Gen.",
    "Gov.",
    "i.e.",
    "I.e.",
    "I.E.",
    "Inc.",
    "Jr.",
    "Ltd.",
    "Md.",
    "Messrs.",
    "Mo.",
    "Mont.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "p.m.",
    "Ph.D.",
    "Prof.",
    "Rep.",
    "Rev.",
    "Sen.",
    "St.",
    "vs.",
    "v.s.",
]:
    special_cases[orth] = [{ORTH: orth}]


def preProcessing(query):
    queryLower = query.lower()
    if queryLower.startswith('eli5'):
        cutMarker = queryLower.find(' ') + 1
        query = query[cutMarker:]

    nlp2.tokenizer.rules = special_cases
    prefixes = (
            ["§", "%", "=", "—", "–", r"\+(?![0-9])"]
            + LIST_PUNCT
            + LIST_ELLIPSES
            + LIST_QUOTES
            + LIST_CURRENCY
            + LIST_ICONS
    )

    suffixes = (
            LIST_PUNCT
            + LIST_ELLIPSES
            + LIST_QUOTES
            + LIST_ICONS
            + ["'s", "'S", "’s", "’S", "—", "–"]
            + [
                r"(?<=[0-9])\+",
                r"(?<=°[FfCcKk])\.",
                r"(?<=[0-9])(?:{c})".format(c=CURRENCY),
                r"(?<=[0-9])(?:{u})".format(u=UNITS),
                r"(?<=[0-9{al}{e}{p}(?:{q})])\.".format(
                    al=ALPHA_LOWER, e=r"%²\-\+", q=CONCAT_QUOTES, p=PUNCT
                ),
                r"(?<=[{au}][{au}])\.".format(au=ALPHA_UPPER),
            ]
    )

    infixes = (
            LIST_ELLIPSES
            + LIST_ICONS
            + [
                r"(?<=[0-9])[+\-\*^](?=[0-9-])",
                r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                    al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                ),
                r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
                r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            ]
    )

    prefixes_re = compile_prefix_regex(prefixes)
    nlp2.tokenizer.prefix_search = prefixes_re.search

    suffixes_re = compile_suffix_regex(suffixes)
    nlp2.tokenizer.suffix_search = suffixes_re.search

    infix_re = compile_infix_regex(infixes)
    nlp2.tokenizer.infix_finditer = infix_re.finditer

    query = query.replace('\n', ' ')
    query = query.replace('\t', ' ')
    query = re.sub(r'(\w\w)\?(\w\w)', r'\1 ? \2', query)
    query = query.replace('(', ' ( ')
    query = query.replace(')', ' ) ')
    query = query.replace('   ', ' ')
    query = query.replace('  ', ' ')

    doc = nlp2(query)  # , disable=['parser', 'ner'])
    tokens = []
    for token in doc:
        if token.text != ' ':
            tokens.append(token.text)

    if len(tokens) == 0:
        print("Zero token sentence detected!")
    return tokens

def giveSpanList(row, tokens, string1, data_type):
    if data_type != 'old':
        string_all = []
        flag = 0
        if row['post_id'] in ['10510109_gab', '1081573659137830912_twitter', '1119979940080783360_twitter']:
            flag = 1
        for exp in string1:
            start, end = int(exp.split('-')[1]), int(exp.split('-')[2])
            if flag == 1:
                print(exp)

            start_pos = 0
            end_pos = 0
            pos = 0
            count = 0
            for tok in tokens:
                if flag == 1:
                    print(count)
                    print(pos)
                if count == start:
                    start_pos = pos
                pos += len(tok) + 1
                if (count + 1) == end:
                    end_pos = pos
                    break

                count += 1

            string_all.append((exp.split('-')[0], start_pos, end_pos))
    else:
        if string1 in ["{}", "{", "}"]:
            return []
        list1 = string1.split("||")
        string_all = []
        for l in list1:
            # collect the string
            # colect the string postion (start--end) in the original text
            string_mask = re.findall('\((.*?)\)', l)[0]
            string = l[len(string_mask) + 2:]
            [start, end] = string_mask.split("--")
            string_all.append((string, start, end))

    return string_all


def returnMask(row, params, tokenizer):
    text_tokens = row['text']

    ##### a very rare corner case
    if len(text_tokens) == 0:
        text_tokens = ['dummy']
        print("length of text ==0")
    #####

    mask_all = row['rationales']
    mask_all_temp = mask_all
    while len(mask_all_temp) != 3:
        mask_all_temp.append([0] * len(text_tokens))

    word_mask_all = []
    word_tokens_all = []

    for mask in mask_all_temp:
        if mask[0] == -1:
            mask = [0] * len(mask)

        list_pos = []
        mask_pos = []

        flag = 0
        for i in range(0, len(mask)):
            if i == 0 and mask[i] == 0:
                list_pos.append(0)
                mask_pos.append(0)
            if flag == 0 and mask[i] == 1:
                mask_pos.append(1)
                list_pos.append(i)
                flag = 1

            elif flag == 1 and mask[i] == 0:
                flag = 0
                mask_pos.append(0)
                list_pos.append(i)
        if list_pos[-1] != len(mask):
            list_pos.append(len(mask))
            mask_pos.append(0)
        string_parts = []
        for i in range(len(list_pos) - 1):
            string_parts.append(text_tokens[list_pos[i]:list_pos[i + 1]])

        word_tokens = [101]
        word_mask = [0]
        for i in range(0, len(string_parts)):
            tokens = ek_extra_preprocess(" ".join(string_parts[i]), params, tokenizer)
            masks = [mask_pos[i]] * len(tokens)
            word_tokens += tokens
            word_mask += masks

        word_tokens = word_tokens[0:(int(params['max_length']) - 2)]
        word_mask = word_mask[0:(int(params['max_length']) - 2)]
        word_tokens.append(102)
        word_mask.append(0)
        word_mask_all.append(word_mask)
        word_tokens_all.append(word_tokens)
    if len(mask_all) == 0:
        word_mask_all = []
    else:
        word_mask_all = word_mask_all[0:len(mask_all)]
    return word_tokens_all[0], word_mask_all