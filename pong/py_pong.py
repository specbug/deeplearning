#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import pygame
pygame.init()


# In[25]:


WIDTH = 1200
HEIGHT = 600
THICKNESS = 30
BALL_RADIUS = 20
PAD_WIDTH = 30
PAD_HEIGHT = 120
VELOCITY = 1
FRAMERATE = 150
BUFFER = 5
AI = True

bgColor = pygame.Color('white')
wallColor = pygame.Color('gray')
padColor = pygame.Color('orange')
ballColor = pygame.Color('red')


# In[26]:


screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen.fill(bgColor)


# In[27]:


class Ball:
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        
    def show(self, color):
        global screen
        pygame.draw.circle(screen, color, (self.x, self.y), BALL_RADIUS)
    
    def level(self, velocity):
        self.vx = velocity if self.vx > 0 else -velocity
        self.vy = velocity if self.vy > 0 else -velocity
    
    def update(self):
        global HEIGHT, THICKNESS, BALL_RADIUS, PAD_WIDTH, PAD_WIDTH, bgColor, ballColor, padObject
        
        newX = self.x + self.vx
        newY = self.y + self.vy
        
        if newX < (THICKNESS + BALL_RADIUS) or (newX > (WIDTH-PAD_WIDTH-BUFFER-BALL_RADIUS) and not (newX >= WIDTH-BUFFER-PAD_WIDTH//2) and (((self.y+BALL_RADIUS) >= padObject.y) and ((self.y-BALL_RADIUS) <= (padObject.y + PAD_HEIGHT)))):
            self.vx = -self.vx
        elif newY < (THICKNESS + BALL_RADIUS) or newY > (HEIGHT-BALL_RADIUS-THICKNESS):
            self.vy = - self.vy    
        else:
            self.show(bgColor)

            self.x = self.x + self.vx
            self.y = self.y + self.vy

            self.show(ballColor)
            
class Pad:
    def __init__(self, y):
        self.y = y
        
    def show(self, color):
        global screen, PAD_HEIGHT, PAD_WIDTH, BUFFER
        pygame.draw.rect(screen, color, pygame.Rect((WIDTH-PAD_WIDTH-BUFFER), self.y, PAD_WIDTH, PAD_HEIGHT))
        
    def update(self, position=None):
        global HEIGHT, THICKNESS, padColor, PAD_HEIGHT
        
        if position == None:
            newY = pygame.mouse.get_pos()[1]
        else:
            newY = position
        if newY > THICKNESS and newY < (HEIGHT-THICKNESS-PAD_HEIGHT):
            self.show(bgColor)
            self.y = newY
            self.show(padColor)


# In[28]:


pygame.draw.rect(screen, wallColor, pygame.Rect(0, 0, WIDTH, THICKNESS))
pygame.draw.rect(screen, wallColor, pygame.Rect(0, THICKNESS, THICKNESS, (HEIGHT-THICKNESS)))
pygame.draw.rect(screen, wallColor, pygame.Rect(0, (HEIGHT-THICKNESS), WIDTH, THICKNESS))

init_x = np.random.choice(range((0+THICKNESS+BALL_RADIUS), ((WIDTH-THICKNESS-BALL_RADIUS)+1)))
init_y = np.random.choice(range((0+THICKNESS+BALL_RADIUS), ((HEIGHT-THICKNESS-BALL_RADIUS)+1)))

ballObject = Ball(init_x, init_y, -VELOCITY, -VELOCITY)
ballObject.show(ballColor)

padObject = Pad(HEIGHT//2)
padObject.show(padColor)


# In[29]:


if AI:
    pongAI = pickle.load(open('pong_AI.sav', 'rb'))
    pongAI
    
    features = ['x', 'y', 'vx', 'vy']
    target = 'pad_y'


# In[30]:


if not AI:
    logger_df = pd.DataFrame(columns=['x', 'y', 'vx', 'vy', 'pad_y'])
game_c = 0
clock = pygame.time.Clock()

while True:
    event = pygame.event.poll()
    if event.type == pygame.QUIT:
        pygame.quit()
        break
    
    game_c += 1
    _ = clock.tick(FRAMERATE)
    
    pygame.display.flip()
    
    ballObject.update()
    
    if AI:
        padObject.update(pongAI.predict([[ballObject.x, ballObject.y, ballObject.vx, ballObject.vy]]))
    else:
        padObject.update(None)
        logger_df = logger_df.append({
            'x': ballObject.x,
            'y': ballObject.y,
            'vx': ballObject.vx,
            'vy': ballObject.vy,
            'pad_y': padObject.y
        }, ignore_index=True, sort=False)

    if game_c%(VELOCITY*5000) == 0:
        VELOCITY += 1
        ballObject.level(VELOCITY)
    
if not AI:
    print(logger_df.shape)
    print(logger_df.head())


# In[ ]:

if not AI:
    logger_df.to_parquet('training_data.parquet')


    # In[ ]:


    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score


    # In[ ]:


    def mape(A, F):
        df = pd.DataFrame()
        df['actual'] = A
        df['forecast'] = F
        df['perc_error'] = (df['forecast'] / df['actual']) - 1
        df['perc_error'] = df['perc_error'].apply(lambda x: np.abs(x))
        df.loc[df['perc_error'].isin([np.nan]), 'perc_error'] = 0
        df.drop(df.loc[df['perc_error'].isin([np.inf])].index, inplace=True)
        return df['perc_error'].mean()


    # In[ ]:


    features = ['x', 'y', 'vx', 'vy']
    target = 'pad_y'


    # In[ ]:


    logger_df = pd.read_parquet('training_data.parquet')


    # In[ ]:


    X = logger_df[features].copy()
    y = logger_df[target].copy()

    (X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

    print('Training data:', (X_train.shape, y_train.shape), '\n')

    print('Validation data:', (X_val.shape, y_val.shape), '\n')

    print('Test data:', (X_test.shape, y_test.shape))


    # In[ ]:


    model = KNeighborsRegressor(n_neighbors=3, n_jobs=-1).fit(X_train, y_train)
    print('Training error:', mape(y_train, model.predict(X_train)), '\n')
    print('Validation error:', mape(y_val, model.predict(X_val)), '\n')


    # In[ ]:


    print('Test error:', mape(y_test, model.predict(X_test)), '\n')


    # In[ ]:


    pickle.dump(model, open('pong_AI.sav', 'wb'))

