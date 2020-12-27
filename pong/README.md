# PyPong
Self learning (squash) pong game

Inspired from [this Computerphile series](https://www.youtube.com/playlist?list=PLzH6n4zXuckoUWpzSEpQNW6I8rXIzyi8w)

A simple (squash) pong game, where the AI learns by watching you play (but not in the reinforcement learning sense, this is achieved using regression).

<img src="https://github.com/specbug/PyPong/blob/master/pongAI.gif"/>


## Installation

Make sure you have [PyGame](https://www.pygame.org/) installed.

`pip install pygame`

In the code repo, set the <b>AI</b> flag to `False` to play the game yourself. Play for a few minutes to let the model train on your playing data. You can then set the <b>AI</b> flag to `True` and watch the model go!!  
Also, don't lose while playing, as that would have a counter effect on the model! (needs improvement)

## Note

This is not a perfect package, it works but is not very effecient or smooth. Feel free to contribute.
