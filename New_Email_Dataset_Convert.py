import pandas as pd
import random
import re
import numpy as np

# 1. Load the uploaded dataset
df = pd.read_csv("/Users/naim/Desktop/Spam Email Detection/old_spam_email_dataset.csv")
df['body_final_modified'] = df['Body']  # Start fresh

# 2. OLD tricks (emojis, typos, leet speak, spammy words, etc.)

def add_emojis(text):
    emojis = ['🚀', '💸', '🔥', '🎯', '🌟', '🏆', '💥']
    emoji = random.choice(emojis)
    parts = text.split()
    if parts:
        insert_at = random.randint(0, len(parts))
        parts.insert(insert_at, emoji)
    return ' '.join(parts)

def random_capitalization(text):
    return ''.join(c.upper() if random.random() < 0.2 else c for c in text)

def introduce_typos(text):
    typos = {'receive': 'recieve', 'money': 'mony', 'free': 'fre', 'account': 'acount', 'click': 'clik'}
    for correct, typo in typos.items():
        text = re.sub(r'\b' + correct + r'\b', typo, text, flags=re.IGNORECASE)
    return text

def add_spammy_keywords(text):
    spammy_words = ['winner', 'urgent', 'claim now', 'limited offer', 'act fast']
    insertion = random.choice(spammy_words)
    parts = text.split()
    if parts:
        insert_at = random.randint(0, len(parts))
        parts.insert(insert_at, insertion.upper())
    return ' '.join(parts)

def leet_speak(text):
    replacements = {'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '$'}
    return ''.join(replacements.get(c.lower(), c) for c in text)

def apply_old_tricks(text):
    text = add_emojis(text)
    text = random_capitalization(text)
    text = introduce_typos(text)
    text = add_spammy_keywords(text)
    text = leet_speak(text)
    return text

# 3. NEW tricks (obfuscated links, unicode, fake personalization, AI buzzwords)

def obfuscate_urls(text):
    return re.sub(r'http[s]?://\S+', lambda m: m.group(0).replace('http', 'hxxp').replace('.', '[.]'), text)

def unicode_spoof(text):
    unicode_map = {'e': 'е', 'a': 'а', 'o': 'о', 'i': 'і', 'c': 'с'}
    return ''.join(unicode_map.get(c, c) for c in text)

def add_ai_buzzwords(text):
    buzzwords = ['synergistic paradigms', 'quantum solutions', 'next-gen optimization', 'AI-driven scalability']
    buzzword = random.choice(buzzwords)
    parts = text.split()
    if parts:
        insert_at = random.randint(0, len(parts))
        parts.insert(insert_at, buzzword)
    return ' '.join(parts)

def fake_personalization(text):
    names = ['John', 'Emma', 'Liam', 'Olivia']
    return f"Hey {random.choice(names)}, {text}"

def apply_newest_tricks(text):
    text = fake_personalization(text)
    text = obfuscate_urls(text)
    text = unicode_spoof(text)
    text = add_ai_buzzwords(text)
    return text

# 4. Splitting the dataset
np.random.seed(42)
indices = np.arange(len(df))
np.random.shuffle(indices)

n_total = len(df)
n_emojis = int(n_total * 0.4)  # 35% add only emojis
n_new = int(n_total * 0.25)     # 30% newest tricks
n_old = n_total - n_emojis - n_new  # Remaining 10% keep old

emojis_idx = indices[:n_emojis]
new_tricks_idx = indices[n_emojis:n_emojis+n_new]
old_idx = indices[n_emojis+n_new:]

# 5. Apply transformations
def add_only_emojis(text):
    return add_emojis(text)

df.loc[emojis_idx, 'body_final_modified'] = df.loc[emojis_idx, 'Body'].apply(add_only_emojis)
df.loc[new_tricks_idx, 'body_final_modified'] = df.loc[new_tricks_idx, 'Body'].apply(apply_newest_tricks)
# old_idx data remains unchanged

# Correct it like this:
df.loc[old_idx, 'body_final_modified'] = df.loc[old_idx, 'body_final_modified'].apply(apply_old_tricks)

# 6. Save the modified dataset
df.to_csv("/Users/naim/Desktop/Spam Email Detection/new_spam_email_dataset.csv", index=False)


