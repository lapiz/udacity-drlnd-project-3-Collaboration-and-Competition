import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque

class Scores():
    def __init__( self, expect, size = 100, check_solved = True ):
        self.window_size = size
        self.scores = []
        self.scores_window = deque(maxlen=size)
        self.scores_log = []
        self.expected = expect
        self.episode = 1
        self.start = datetime.now()
        self.check_solved = check_solved

    def AddScore( self, score ):
        self.scores.append(score)
        self.scores_window.append(score)

        window_mean = np.mean(self.scores_window)
        print(F'\r[{datetime.now()}] Episode {self.episode}\tScore: {score:.2f}\tAverage Score: {window_mean:.2f}')
        if self.episode % self.window_size == 0:
            self.scores_log.append( (self.episode, window_mean) )
        if window_mean >= self.expected:
            if self.check_solved:
                print(F'\n[{datetime.now()}] Environment solved in {self.episode-self.window_size:d} episodes!\tAverage Score: {window_mean:.2f}')
                self.scores_log.append( (self.episode, window_mean) )
            return True

        self.episode += 1
        return False

    def FlushLog(self, prefix, showNow):
        print( F'\nMin: {min(self.scores)}')
        print( F'Max: {max(self.scores)}')
        print( F'Count: {len(self.scores)}')
        print( F'Avg: {np.mean(self.scores)}')
        print( F'Std: {np.std(self.scores)}')
        print( 'Done\n\n' )
        fig = plt.figure()
        plt.plot(np.arange(len(self.scores)), self.scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig(F'{prefix}_scores.png', dpi=300)
        if showNow:
            plt.show()

        window_mean = np.mean(self.scores_window)
        self.scores_log.append( (self.episode, window_mean) )
        with open(F'{prefix}_log.txt', 'w') as f:
            f.write( F'Job started at {self.start}\n' )
            f.writelines( [ F'Episode {log[0]}\tAverage Score: {log[1]:.2f}\n' for log in self.scores_log ] )
            f.write( F'Min: {min(self.scores)}\n')
            f.write( F'Max: {max(self.scores)}\n')
            f.write( F'Count: {len(self.scores)}\n')
            f.write( F'Avg: {np.mean(self.scores)}\n')
            f.write( F'Std: {np.std(self.scores)}\n')
            f.write( F'Job finished at {datetime.now()}\n' )
            f.flush()
            f.close()
