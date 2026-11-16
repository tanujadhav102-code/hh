
import random
import pickle

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9

    def display(self):
        print("\n")
        for i in range(0, 9, 3):
            print("|".join(self.board[i:i+3]))
            if i < 6:
                print("-----")

    def available_moves(self):
        return [i for i, x in enumerate(self.board) if x == ' ']

    def make_move(self, position, player):
        if position in self.available_moves():
            self.board[position] = player
            return True
        return False

    def check_winner(self):
        wins = [
            (0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8),
            (0,4,8), (2,4,6)
        ]

        for a, b, c in wins:
            if self.board[a] == self.board[b] == self.board[c] != ' ':
                return self.board[a]

        return None


class QLearningAgent:
    def __init__(self):
        self.q = {}
        self.lr = 0.1
        self.discount = 0.9
        self.epsilon = 0.2

    def get_state(self, board):
        return tuple(board)

    def choose_action(self, board):
        state = self.get_state(board)
        moves = [i for i, x in enumerate(board) if x == ' ']

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(moves)

        qvals = [self.q.get((state, a), 0) if a in moves else float('-inf') for a in range(9)]
        return qvals.index(max(qvals))

    def update(self, state, action, reward, next_state, done):
        old_q = self.q.get((state, action), 0)

        if done:
            future_q = 0
        else:
            future_q = max(
                self.q.get((next_state, a), 0)
                for a in range(9)
                if next_state[a] == ' '
            )

        self.q[(state, action)] = old_q + self.lr * (
            reward + self.discount * future_q - old_q
        )


# Training the AI
agent = QLearningAgent()

for episode in range(5000):
    game = TicTacToe()
    done = False

    while not done:
        state = agent.get_state(game.board)
        action = agent.choose_action(game.board)

        game.make_move(action, 'X')

        winner = game.check_winner()

        if winner or not game.available_moves():
            agent.update(state, action, 1 if winner == 'X' else 0, agent.get_state(game.board), True)
            break

        opponent_move = random.choice(game.available_moves())
        game.make_move(opponent_move, 'O')

        winner = game.check_winner()
        done = winner is not None or not game.available_moves()

        reward = -1 if winner == 'O' else 0
        agent.update(state, action, reward, agent.get_state(game.board), done)

    agent.epsilon = max(0.01, agent.epsilon * 0.999)


# Save Q-table
with open("q_table.pkl", "wb") as file:
    pickle.dump(agent.q, file)

print("Training completed!")
print("Q-table saved successfully!")


# Play the game
game = TicTacToe()

print("\nWelcome to Tic Tac Toe!")
print("You are O and AI is X")

while True:
    game.display()

    action = agent.choose_action(game.board)
    game.make_move(action, 'X')

    if game.check_winner() == 'X':
        game.display()
        print("AI wins!")
        break

    if not game.available_moves():
        game.display()
        print("Match Draw!")
        break

    while True:
        try:
            position = int(input("Enter position (0-8): "))
            if game.make_move(position, 'O'):
                break
            print("Position already occupied. Try again.")
        except ValueError:
            print("Please enter a valid number.")

    if game.check_winner() == 'O':
        game.display()
        print("Congratulations! You win!")
        break

    if not game.available_moves():
        game.display()
        print("Match Draw!")
        break
