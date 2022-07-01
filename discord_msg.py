import os
import discord
from dotenv import load_dotenv
import random

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
        "gaali mat do bhai",
        "please no gaali!",
        "i took offense"
    ]
    bad = 'sala'
    if bad in message.content:
        response = random.choice(ban_messages)
        await message.channel.send(response)

client.run(TOKEN)