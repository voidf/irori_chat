from mongoengine import Document
from mongoengine.fields import StringField
from mongoengine import connect
from asyncable import Asyncable
import asyncio
connect(host='mongodb://localhost:27017/irori_chat')
# {q:{$regex:'[^0-9^a-z^A-Z^\\s]'},a:{$regex:'^[0-9\\.\\?!@a-zA-Z]+$'}}
# {$expr:{$lt:[{$strLenCP:'$a'}, 3]}, a:{$regex: '[a-zA-Z]'}}
class Conversation(Document, Asyncable):
    q = StringField()
    a = StringField()

# res = Conversation.objects(a='= =').delete()
# print(res)

res = Conversation.objects(q__contains='鸡')

for i in res:
    print(i.q, i.a, sep='\t\t')
    cmd = None
    while not cmd:
        cmd = input('delete?(y/N)')
    if cmd == 'y':
        i.delete()
    
    


print(len(res))
