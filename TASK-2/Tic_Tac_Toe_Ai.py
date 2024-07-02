from tkinter import *
import random

# Initialize the main window
root=Tk()
root.title("Tic Tac Toe")
root.geometry("500x650")
root.resizable(0,0)

# Frames
# Each and every Frame display row, column , grid and information
frame1 = Frame(root)
frame1.pack()
title = Label(frame1, text="Tic Tac Toe", font=("Times New Roman", 36), fg="brown")
title.pack()

OptionFrame = Frame(root, bg="grey")
OptionFrame.pack()

frame2 = Frame(root)
frame2.pack()

frame3 = Frame(root)
frame3.pack()

# Variables used to play Game
# Initializing the Board with Empty Values, by giving 1 to 9 keys 
# 1 to 9 elements represents the boxes, buttons names
Board = [" "] * 9 
turn = "X"
game_end = False
mode = "Singleplayer"
win_label = None
draw_label = None

# Modes of the game, Singleplayer or Multiplayer
# By default Game starts as Singleplayer or Computer VS User
def changeModeToSingleplayer():             # Changing Mode from Multiplayer to Singleplayer
    global mode
    mode = "Singleplayer"
    # Changing Button colors to show Singleplayer mode is activated
    MultiplayerButton["bg"] = "pink"
    SingleplayerButton["bg"] = "red"


def changeModeToMultiplayer():              # Changing Mode from Singleplayer to Multiplayer
    global mode
    mode = "Multiplayer"
    # Changing Button colors to show Multiplayer mode is activated
    SingleplayerButton["bg"] = "pink"
    MultiplayerButton["bg"] = "orange"


# Function to update the board
def UpdateBoard():
    for i in range(9):
        all_buttons[i]["text"] = Board[i]
        if Board[i] == "X":
            all_buttons[i]["fg"] = "red"
        elif Board[i] == "O":
            all_buttons[i]["fg"] = "blue"


# Function to check for a winner
def WinnerCheck(player):
    win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                      (0, 3, 6), (1, 4, 7), (2, 5, 8),
                      (0, 4, 8), (2, 4, 6)]
    for condition in win_conditions:
        if Board[condition[0]] == Board[condition[1]] == Board[condition[2]] == player:
            return True
    return False


# Function to check for a draw
def DrawCheck():
    return " " not in Board
    

# Function to restart the game
def restartGame():
    global game_end, win_label, draw_label, turn
    game_end = False
    for i in range(9):
        Board[i] = " "
        
    if win_label:
        win_label.destroy()
    if draw_label:
        draw_label.destroy()

    turn = "X"
    UpdateBoard()

        
# Minimax algorithm for the computer move
def minimax(Board, depth, isMaximizing):
    if WinnerCheck("O"):
        return 1
    if WinnerCheck("X"):
        return -1
    if DrawCheck():
        return 0
    
    if isMaximizing:
        bestScore = -float("inf")
        for i in range(9):
            if(Board[i]==" "):
                Board[i] = "O"
                score = minimax(Board, depth+1, False)
                Board[i] = " "
                bestScore = max(score, bestScore)
        return bestScore
    else:
        bestScore = float("inf")
        for i in range(9):
            if(Board[i]==" "):
                Board[i] = "X"
                score = minimax(Board, depth+1, True)
                Board[i] = " "
                bestScore = min(score, bestScore)
        return bestScore

# Function to handle the computer's move
def playComputer():
    bestScore = -float("inf")
    bestMove = None

    for i in range(9):
        if(Board[i]==" "):
            Board[i] = "O"
            score = minimax(Board, 0, False)
            Board[i] = " "
            if score > bestScore:
                bestScore = score
                bestMove = i
    if bestMove is not None:
        Board[bestMove] = "O"


# Function to handle a User's move
def play(event):
    global turn, game_end, win_label, draw_label 
    if game_end:
        return             # Returns the value None
    
    button = event.widget
    index = all_buttons.index(button)
    if Board[index] == " ":
        Board[index] = turn
        UpdateBoard()
        if WinnerCheck(turn):
            win_label = Label(frame2, text=f"{turn} Wins the Game", bg="lightgreen", font=("Montserrat", 27))
            win_label.grid(row=4, column=0, columnspan=3, pady=5)
            game_end = True
        elif DrawCheck():
            draw_label = Label(frame2, text=f"Game Draw", bg="yellow", font=("Montserrat", 25))
            draw_label.grid(row=4, column=0, columnspan=3, pady=5)
            game_end = True

        else:
            turn = "O" if turn == "X" else "X"
            if mode == "Singleplayer" and turn == "O" and not game_end:
                playComputer()
                if WinnerCheck(turn):
                    win_label = Label(frame2, text=f"{turn} Wins the Game", bg="lightgreen", font=("Montserrat", 27))
                    win_label.grid(row=4, column=0, columnspan=3, pady=5)
                    game_end = True
                elif DrawCheck():
                    draw_label = Label(frame2, text=f"Game Draw", bg="yellow", font=("Montserrat", 25))
                    draw_label.grid(row=4, column=0, columnspan=3, pady=5)
                    game_end = True

                turn = "X"     
                UpdateBoard()
    

# Buttons to select the mode
# SinglePlayer or VS Computer Button
SingleplayerButton = Button(OptionFrame, text="VS Computer", width=12, height=1, font=("Lora", 15), relief=RAISED , borderwidth=3, background="pink", command=changeModeToSingleplayer)
SingleplayerButton.grid(row=0, column=0, columnspan=1, sticky=NW, pady=5)

# Multiplayer Button
MultiplayerButton = Button(OptionFrame, text="2 Player", width=12, height=1, font=("Lora", 15), relief=RAISED , borderwidth=3, background="pink", command=changeModeToMultiplayer)
MultiplayerButton.grid(row=0, column=1, columnspan=1, sticky=NW, pady=5)


# Creating the 3x3 Grid or Matrix of buttons
all_buttons = []
for row in range(3):
    for col in range(3):
        button = Button(frame2, text=" ", width=4, height=2, font=("Arial", 30), relief=RAISED, borderwidth=5, background="white")
        button.grid(row=row, column=col)
        button.bind("<Button-1>", play)
        all_buttons.append(button)

# Restart game button
reset_button = Button(frame3, text="Restart Game", width=12, height=1, font=("Merriweather", 20), relief=RAISED , borderwidth=5, bg="cyan", command=restartGame)
reset_button.grid(row=5, column=0, columnspan=3, pady=10)

root.mainloop()