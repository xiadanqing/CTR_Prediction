import dataprocess
import pandas as pd


def user_process(k):
    user = pd.read_csv(dataprocess.data_path + 'user.csv')
    user['new_tag'] = user['tag'].str.cat(user['outertag'], sep='|')
    user['new_guid'] = user['guid'].str.cat(user['deviceid'], sep=',', na_rep='-')
    def tag2dic(line):
        l = line.split('|')
        c0 = []
        c1 = []
        for x in l:
            cache = x.split(':')
            if len(cache) == 2:
                c0.append(cache[0])
                c1.append(cache[1])
        return dict(zip(c0, c1))

    user.loc[user['new_tag'].isna() == False, 'new_tag'] = list(
        map(tag2dic, user.loc[user['new_tag'].isna() == False, 'new_tag']))
    user.loc[user['new_tag'].isna() == False, 'new_tag'] = user.loc[user['new_tag'].isna() == False, 'new_tag'].apply(
        lambda x: dataprocess.topKeys(x, k))
    user['new_tag'] = user['new_tag'].fillna('0')
    user.loc[user['new_tag'].isna() == False, 'new_tag'] = list(
        map(dataprocess.varFeatEncoder, user.loc[user['new_tag'].isna() == False, 'new_tag']))
    user['guid'] = user['new_guid'].apply(lambda x: dataprocess.catfunc(x))
    user.drop(['tag', 'outertag', 'new_guid', 'deviceid'], axis=1, inplace=True)
    user = user.drop_duplicates('guid')

    tags = user.loc[user['new_tag'].isna() == False, 'new_tag'].tolist()
    tags = [i for p in tags for i in p]
    tags_nums = len(set(tags))

    del tags
    return user, tags_nums

