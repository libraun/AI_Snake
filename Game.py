import pygame as pg
import numpy as np
import random

random.seed(13.145)

from typing import Tuple

import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F

from collections import deque




HEIGHT = 25
LENGTH = 25 

MAX_MEMORY = 100_000
DIRECTIONS = {
    (1, 0) : {
        "FORWARD" : (1,0),
        "LEFT" : (0,-1),
        "RIGHT" : (0,1)
    },
    (-1,0) : {
        "FORWARD" : (-1,0),
        "LEFT" : (0,-1),
        "RIGHT" : (0,1)
    },
    (0, 1) : {
        "FORWARD" : (0,1),
        "LEFT" : (-1,0),
        "RIGHT" : (1,0)
    },
    (0, -1) : {
        "FORWARD" : (0,-1),
        "LEFT" : (-1,0),
        "RIGHT" : (1,0)
    },
}        

class Board:
    def __init__(self):
          

        self.reset()
    def reset(self) -> None:
        init_snake_head = (INIT_POSITION[0], INIT_POSITION[1] + 1)
        self.snake_body = [INIT_POSITION, init_snake_head]

        self.last_direction = (0,1)

        self.score = 0

        self.grid = np.zeros((HEIGHT,LENGTH),dtype=float)
        self.generate_fruit()
        for segment in self.snake_body[:-2]:
            pos_x, pos_y = segment[0], segment[1]
            self.grid[pos_x][pos_y] = 1
        self.grid[self.snake_body[-1][0]][self.snake_body[-1][1]] = 3

        
        self.set_state(0,0)

        #self.state = np.zeros((9,),dtype=float)
    def generate_fruit(self ) -> None:
        self.fruit = (random.randrange(0,HEIGHT), \
                          random.randrange(0,LENGTH))
        while self.fruit not in self.snake_body:
            self.fruit = (random.randrange(0,HEIGHT), \
                          random.randrange(0,LENGTH))
        self.grid[self.fruit[0]][self.fruit[1]] = 2

    def move_snake(self, direction: str) -> tuple:

        new_head_pos = self.get_coordinates(direction)
        if new_head_pos == None:

            self.set_state(-10,1)

            return (-10,self.score,1)
        
        last_head_pos = self.snake_body[-1]
        self.grid[last_head_pos[0]][last_head_pos[1]] = 1
        
        self.snake_body.append(new_head_pos)
        self.grid[new_head_pos[0]][new_head_pos[1]] = 3
        
        for segment in self.snake_body[:-2]:
            self.grid[segment[0]][segment[1]] = 1
        
        if new_head_pos != self.fruit:
            reward = 0
            removed_segment = self.snake_body.pop(0)
            empty_x, empty_y = removed_segment[0],removed_segment[1]

            self.grid[empty_x][empty_y] = 0
        # Snake has reached fruit; regenerate fruit coords
        else:
            reward = 10
            self.generate_fruit()
            
            self.score += 1
        self.set_state(reward,0)
        return (reward,self.score, 0)

    def set_state(self,reward,game_over) -> None:
        fruit_forward = self.get_fruit_forward()
        key_vals = [key for key in DIRECTIONS.keys()]
        state_list = [
            key_vals.index(self.last_direction) == 3,
            key_vals.index(self.last_direction) == 2,
            key_vals.index(self.last_direction) == 1,
            key_vals.index(self.last_direction) == 1,
            fruit_forward,
            self.get_fruit_right(),
            self.get_fruit_left(),
            self.get_danger_at("FORWARD"),
            self.get_danger_at("RIGHT"),
            self.get_danger_at("LEFT"),
        ]
        self.state = np.array(state_list,dtype=float)

    def get_coordinates(self, direction: str) -> Tuple[int] | None:
       # self.last_direction = direction
        rel_pos = DIRECTIONS[self.last_direction][direction]
        
        last_head_pos = self.snake_body[-1]
        
        new_head_x = last_head_pos[0] + rel_pos[0]
        new_head_y = last_head_pos[1] + rel_pos[1]

        new_pos = (new_head_x, new_head_y)
        
        if (new_pos[0] not in range(0,LENGTH) or new_pos[1] not in range(0,LENGTH)) \
            or new_pos in self.snake_body[:-2]:
            
            return None
        
        self.last_direction = rel_pos
        return new_pos

    def get_fruit_forward(self) -> int:

        if self.last_direction[0] != 0:
            if self.fruit[1] != self.snake_body[-1][1]:
                return 0
            elif self.last_direction[0] == 1:
                return int(self.fruit[0] > self.snake_body[-1][0] )
            else:
                return int(self.fruit[0] < self.snake_body[-1][0]) 

        else:
            if self.fruit[0] != self.snake_body[-1][0]:
                return 0
            elif self.last_direction[1] == 1:
                return int(self.fruit[1] > self.snake_body[-1][1] )
            else:
                return int(self.fruit[1] < self.snake_body[-1][1])
            
    def get_fruit_right(self) -> bool:
        if self.last_direction[1] != 0:
            column = self.grid.transpose()[self.snake_body[-1][1]]
            if 2 not in column:
                return False
            i, = np.where(column == 3)
            j, = np.where(column == 2)
            if i.size == 0 or j.size == 0:
                return False
            return (i < j).any()
        else:
            row = self.grid[self.snake_body[-1][0]]
            if 2 not in row:
                return False
            i, = np.where(row == 3)
            j, = np.where(row == 2)
            if j.size == 0 or j.size == 0:
                return False

            return (i > j).any()
        
    def get_fruit_left(self) -> bool:
        if self.last_direction[1] != 0:

            column = self.grid.transpose()[self.snake_body[-1][1]]
            if 2 not in column:
                return False
            i, = np.where(column == 3)
            j, = np.where(column == 2)
            if i.size or j.size:
                return False

            return (i > j).any()
        else:
            row = self.grid[self.snake_body[-1][0]]
            if 2 not in row:
                return False
            i, = np.where(row == 3)
            j, = np.where(row == 2)
            if i.size == 0 or j.size == 0:
                return False

            return (i < j).any()
            
    def get_danger_at(self, direction: str) -> bool:
        head = self.snake_body[-1]
        rel_pos = DIRECTIONS[self.last_direction][direction]
        
        pos_y = head[0] + rel_pos[0]
        pos_x = head[1] + rel_pos[1]

        return pos_x not in range(0,LENGTH) or pos_y not in range(0,LENGTH) \
            or self.grid[pos_x][pos_y] == 1        



INIT_POSITION = (10,10)

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
def drawGrid():
    global SCREEN,BOARD
    SCREEN.fill((0,0,0))
    blockSize = 20 #Set the size of the grid block
    for x in range(0, WINDOW_WIDTH, blockSize):
        for y in range(0, WINDOW_HEIGHT, blockSize):
            rect = pg.Rect(x, y, blockSize, blockSize)
            pg.draw.rect(SCREEN, pg.Color((200,200,200)) , rect,1)
    for body_segment in BOARD.snake_body:
        rect = pg.Rect(body_segment[0] * 20, body_segment[1] * 20,20,20)
        pg.draw.rect(SCREEN, pg.Color((200,200,200)),rect)  
    fruit_rect = (BOARD.fruit[0] * 20, BOARD.fruit[1] * 20, 20, 20)
    pg.draw.rect(SCREEN,pg.Color((255,0,0)),fruit_rect)

class Trainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(),lr=self.lr)
        self.criterion = nn.MSELoss()
   
    def train_step(self,state,action,reward,next_state,done):
        state = torch.tensor(state,dtype=torch.float)
        next_state = torch.tensor(next_state,dtype=torch.float)
        action = torch.tensor(action,dtype=torch.float)
        reward = torch.tensor(reward,dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new
        self.optimizer.zero_grad()

        loss = self.criterion(target,pred)
        loss.backward()

        self.optimizer.step()


class Agent:

    def __init__(self,model, trainer: Trainer):
        self.model = model
        self.memory = deque(maxlen=MAX_MEMORY)
        self.trainer = trainer
        self.n_games = 0

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
    def train_step(self,state,actions,rewards,next_states,done):
        self.trainer.train_step(state,actions,rewards,next_states,done)
    def train(self):
        if len(self.memory) > 1024:
            mini_sample = random.sample(self.memory,1024)
        else:
            mini_sample = self.memory
        state,actions,rewards,next_states,done = zip(*mini_sample)
        self.trainer.train_step(state,actions,rewards,next_states,done)

    def get_action(self,state):
        final_move = [0,0,0]
        self.epsilon = 80 - self.n_games
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state_vec = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state_vec)

            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move




class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10,256)
        self.linear2 = nn.Linear(256,3)
     #   self.dropout = nn.Dropout(p=0.1)
    def forward(self,x):
        x = F.relu(self.linear(x))
       ## x = self.dropout(x)
        x = self.linear2(x)
        return x

DIRECTION_NAMES = ("FORWARD","LEFT","RIGHT")
if __name__ == "__main__":

    BOARD = Board()
    
    pg.init()

    SCREEN = pg.display.set_mode((500, 500))
    clock = pg.time.Clock()

    drawGrid()

    running = True
    model = Model()
    model.load_state_dict(torch.load("model4.pth"))
    trainer = Trainer(model,lr=0.001,gamma=0.99)
    agent = Agent(model,trainer=trainer)

    num_steps = 0
    total_score = 0

    mean_score = 0
    while running:
        state_old = BOARD.state
        final_action = agent.get_action(BOARD.state)

        move_vector = DIRECTION_NAMES[final_action.index(1)]

        reward,score,done = BOARD.move_snake(move_vector) 

        next_state = BOARD.state
        num_steps += 1
        if num_steps >= 1000 and score < 50:
            reward = -5
            done = 1
        agent.train_step(state_old,final_action,reward,next_state,done)
        agent.remember(state_old,final_action,reward,next_state,done)
        if done:
            agent.n_games += 1
            total_score += score
            mean_score = total_score / agent.n_games
            print("Score: ",score, "; Num games: ", agent.n_games, " ; Mean score: ", mean_score)
            num_steps = 0
            BOARD.reset()
            agent.train()
        drawGrid()
        pg.display.flip()
        clock.tick(200)