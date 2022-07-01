import os

import random
import classifier
import transliterate
import nsfw_text


import discord
from dotenv import load_dotenv

load_dotenv('toke.env')
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
client = discord.Client()
@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord and is ready to BAN unprofessional USERS!!!')
    print(f'{client.user} has connected to '+GUILD)

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    ban_messages = [
        "No use of swear words please",
        "please dont use hate speech!",
        "Hate speech detected"
    ]
    initial_list = message.content.split(" ")
    final_message=transliterate.op_bhai(message.content)
    final_list = final_message.split(" ")
    m = classifier.preprocessing(final_list)
    z = classifier.tokenize(m, 0)
    arr = classifier.model_over.predict(z)
    print(initial_list)
    print(final_list)
    print()
    flag = 0
    swear_words = nsfw_text.swear()
    list3 = set(swear_words) & set(initial_list)
    list4 = sorted(list3, key=lambda k: swear_words.index(k))
    if len(list4) > 0:
        flag = 1
    if 0 in arr:
        flag = 1
    if 1 in arr:
        flag = 1
    if flag == 1:
        response = random.choice(ban_messages)
        await message.channel.send(response)
client.run(TOKEN)