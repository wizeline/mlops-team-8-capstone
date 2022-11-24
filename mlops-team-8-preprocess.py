import boto3
import pandas as pd
import regex as re

pd.set_option('display.max_columns', None)
s3 = boto3.resource('s3')
bucket=s3.Bucket('mlops-team-8')
directory = 'maildir/'

all_files = []
for object_summary in bucket.objects.filter(Prefix=directory):
    all_files.append(object_summary.key)


def parse_email(email_file, filename):
    fields = ['Message-ID','Date','From','To','Subject','X-FileName']
    f = email_file.decode('windows-1252')
    lines = f.split('\n')
    email={}
    body=''
    field = ''
    last_field = ''
    value = ''
    email['filename'] = filename
    email['username'] = filename.split('/')[1]
    for line in lines:
        for field in fields:
            if (field in line) & (last_field != 'X-FileName'):
                pair = line.split(':',1)
                if pair[0] != field:
                    break
                key = pair[0].lower()
                value = pair[1].strip()
                email[key] = value
                last_field = field
                break
            else:
                if last_field == 'X-FileName':
                    body += ' ' + line.strip()
                    email['body'] = body
                    break
                elif ':' in line:
                    if field == 'X-FileName':
                        pair = line.split(':',1)
                        if pair[0] != field:
                            break
                        key = pair[0].lower()
                        value = pair[1].strip()
                        email[key] = value
                        last_field = field
                        break
                    else:
                        continue
                else:
                    value += line
                    email[key] = value
                    break
    return email
stop = set([
    'a',
    'about',
    'above',
    'across',
    'after',
    'afterwards',
    'again',
    'against',
    'ain',
    'all',
    'almost',
    'alone',
    'along',
    'already',
    'also',
    'although',
    'always',
    'am',
    'among',
    'amongst',
    'amoungst',
    'amount',
    'an',
    'and',
    'another',
    'any',
    'anyhow',
    'anyone',
    'anything',
    'anyway',
    'anywhere',
    'are',
    'aren',
    'around',
    'as',
    'at',
    'back',
    'be',
    'became',
    'because',
    'become',
    'becomes',
    'becoming',
    'been',
    'before',
    'beforehand',
    'behind',
    'being',
    'below',
    'beside',
    'besides',
    'between',
    'beyond',
    'bill',
    'both',
    'bottom',
    'but',
    'by',
    'call',
    'can',
    'cannot',
    'cant',
    'co',
    'con',
    'could',
    'couldn',
    'couldnt',
    'cry',
    'd',
    'de',
    'describe',
    'detail',
    'did',
    'didn',
    'do',
    'does',
    'doesn',
    'doing',
    'don',
    'done',
    'down',
    'due',
    'during',
    'each',
    'eg',
    'eight',
    'either',
    'eleven',
    'else',
    'elsewhere',
    'empty',
    'enough',
    'etc',
    'even',
    'ever',
    'every',
    'everyone',
    'everything',
    'everywhere',
    'except',
    'few',
    'fifteen',
    'fify',
    'fill',
    'find',
    'fire',
    'first',
    'five',
    'for',
    'former',
    'formerly',
    'forty',
    'found',
    'four',
    'from',
    'front',
    'full',
    'further',
    'get',
    'give',
    'go',
    'had',
    'hadn',
    'has',
    'hasn',
    'hasnt',
    'have',
    'haven',
    'having',
    'he',
    'hence',
    'her',
    'here',
    'hereafter',
    'hereby',
    'herein',
    'hereupon',
    'hers',
    'herself',
    'him',
    'himself',
    'his',
    'how',
    'however',
    'hundred',
    'i',
    'ie',
    'if',
    'in',
    'inc',
    'indeed',
    'interest',
    'into',
    'is',
    'isn',
    'it',
    'its',
    'itself',
    'just',
    'keep',
    'last',
    'latter',
    'latterly',
    'least',
    'less',
    'll',
    'ltd',
    'm',
    'ma',
    'made',
    'many',
    'may',
    'me',
    'meanwhile',
    'might',
    'mightn',
    'mill',
    'mine',
    'more',
    'moreover',
    'most',
    'mostly',
    'move',
    'much',
    'must',
    'mustn',
    'my',
    'myself',
    'name',
    'namely',
    'needn',
    'neither',
    'never',
    'nevertheless',
    'next',
    'nine',
    'no',
    'nobody',
    'none',
    'noone',
    'nor',
    'not',
    'nothing',
    'now',
    'nowhere',
    'o',
    'of',
    'off',
    'often',
    'on',
    'once',
    'one',
    'only',
    'onto',
    'or',
    'other',
    'others',
    'otherwise',
    'our',
    'ours',
    'ourselves',
    'out',
    'over',
    'own',
    'part',
    'per',
    'perhaps',
    'please',
    'put',
    'rather',
    're',
    's',
    'same',
    'see',
    'seem',
    'seemed',
    'seeming',
    'seems',
    'serious',
    'several',
    'shan',
    'she',
    'should',
    'shouldn',
    'show',
    'side',
    'since',
    'sincere',
    'six',
    'sixty',
    'so',
    'some',
    'somehow',
    'someone',
    'something',
    'sometime',
    'sometimes',
    'somewhere',
    'still',
    'such',
    'system',
    't',
    'take',
    'ten',
    'than',
    'that',
    'the',
    'their',
    'theirs',
    'them',
    'themselves',
    'then',
    'thence',
    'there',
    'thereafter',
    'thereby',
    'therefore',
    'therein',
    'thereupon',
    'these',
    'they',
    'thick',
    'thin',
    'third',
    'this',
    'those',
    'though',
    'three',
    'through',
    'throughout',
    'thru',
    'thus',
    'to',
    'together',
    'too',
    'top',
    'toward',
    'towards',
    'twelve',
    'twenty',
    'two',
    'un',
    'under',
    'until',
    'up',
    'upon',
    'us',
    've',
    'very',
    'via',
    'was',
    'wasn',
    'we',
    'well',
    'were',
    'weren',
    'what',
    'whatever',
    'when',
    'whence',
    'whenever',
    'where',
    'whereafter',
    'whereas',
    'whereby',
    'wherein',
    'whereupon',
    'wherever',
    'whether',
    'which',
    'while',
    'whither',
    'who',
    'whoever',
    'whole',
    'whom',
    'whose',
    'why',
    'will',
    'with',
    'within',
    'without',
    'won',
    'would',
    'wouldn',
    'y',
    'yet',
    'you',
    'your',
    'yours',
    'yourself',
    'yourselves'
])
def clean_text(data):
    data = str(data)
    if data is not None:
        data = ' '.join([word for word in data.split() if word not in stop])
        data = data.lower()
        data = re.sub('\W+',' ', data)
        return data.strip()
    return ''
email_list = []
for email_file in all_files:
    if email_file.split('/')[-1] == '.DS_Store':
        continue
    obj = s3.Object(bucket.name, email_file)
    f = obj.get()['Body'].read()
    email_list.append(parse_email(f, email_file))
    #f.close()
    
email_df = pd.DataFrame(email_list)
email_df = email_df.reset_index(drop=True)
email_df['body_cleansed'] = email_df["body"].apply(clean_text)

for i in email_df["username"].unique():
    email_df[email_df["username"]==i].to_parquet(f's3://{bucket.name}/maildir-stg/emails_{i}.parquet')
