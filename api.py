from config import *
import torch
from depend.funcs import initialize, evaluate
from prework import fil
import pickle

from fastapi import FastAPI
def preload() -> FastAPI():
    with open('processed_voc_pairs', 'rb') as f:
        voc, pairs = pickle.load(f)
    app = FastAPI()
    encoder, decoder, searcher = initialize(voc, pairs)
    app.encoder = encoder
    app.decoder = decoder
    app.searcher = searcher
    app.voc = voc
    app.pairs = pairs
    return app

app = preload()

@app.get('/')
async def predict(word: str):
    try:
        reply = evaluate(
            app.encoder, 
            app.decoder,
            app.searcher,
            app.voc,
            fil(word)
        )
        rep = []
        for i in reply:
            if i == EOS_TOKEN:
                break
            elif i!=PAD_TOKEN:
                rep.append(i)
        reply = ''.join(rep)
    except KeyError:
        reply = "Error: Encountered unknown word."


    return {'reply': reply}
