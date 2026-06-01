import tkinter as tk
from tkinter import font as tkfont
import random

BG = "#1a1a2e"
PANEL = "#16213e"
ACCENT = "#e94560"
GREEN = "#0f9b58"
GOLD = "#f5a623"
WHITE = "#f0f0f0"
GRAY = "#888888"
DOOR_CLOSED = "#0f3460"
DOOR_OPEN_CAR = "#f5a623"
DOOR_OPEN_GOAT = "#2a5a2a"


class MontyHallApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Monty Hall Problem Simulator")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self.stats = {"stay_wins": 0, "stay_total": 0, "switch_wins": 0, "switch_total": 0}
        self.car_door = None
        self.player_choice = None
        self.revealed_door = None
        self.game_phase = "choose"  # choose -> revealed -> done

        self._build_ui()
        self._reset_game()

    def _build_ui(self):
        title_frame = tk.Frame(self.root, bg=BG)
        title_frame.pack(pady=(20, 5))

        tk.Label(title_frame, text="MONTY HALL PROBLEM", bg=BG, fg=WHITE,
                 font=("Courier New", 22, "bold")).pack()
        tk.Label(title_frame, text="Three doors. One car. Two goats. Your call.",
                 bg=BG, fg=GRAY, font=("Courier New", 10)).pack(pady=(2, 0))

        self.msg_var = tk.StringVar()
        self.msg_label = tk.Label(self.root, textvariable=self.msg_var,
                                  bg=BG, fg=GOLD, font=("Courier New", 12, "bold"),
                                  wraplength=560, justify="center")
        self.msg_label.pack(pady=(12, 8))

        doors_frame = tk.Frame(self.root, bg=BG)
        doors_frame.pack(pady=5)

        self.door_buttons = []
        self.door_canvases = []
        self.door_labels = []

        for i in range(3):
            col_frame = tk.Frame(doors_frame, bg=BG)
            col_frame.grid(row=0, column=i, padx=18)

            canvas = tk.Canvas(col_frame, width=130, height=180,
                               bg=PANEL, highlightthickness=2,
                               highlightbackground=DOOR_CLOSED)
            canvas.pack()
            self.door_canvases.append(canvas)

            lbl = tk.Label(col_frame, text=f"Door {i+1}", bg=BG, fg=WHITE,
                           font=("Courier New", 11, "bold"))
            lbl.pack(pady=(5, 3))

            btn = tk.Button(col_frame, text="Pick Door",
                            command=lambda idx=i: self._on_door_click(idx),
                            bg=DOOR_CLOSED, fg=WHITE,
                            font=("Courier New", 10, "bold"),
                            relief="flat", padx=10, pady=5,
                            activebackground=ACCENT, activeforeground=WHITE,
                            cursor="hand2")
            btn.pack()
            self.door_buttons.append(btn)
            self.door_labels.append(lbl)

        # Stats panel
        stats_outer = tk.Frame(self.root, bg=BG)
        stats_outer.pack(pady=(18, 5))

        tk.Label(stats_outer, text="-- YOUR STATS --", bg=BG, fg=GRAY,
                 font=("Courier New", 9)).pack()

        stats_row = tk.Frame(stats_outer, bg=BG)
        stats_row.pack(pady=(6, 0))

        stay_frame = tk.Frame(stats_row, bg=PANEL, padx=18, pady=10)
        stay_frame.grid(row=0, column=0, padx=14)
        tk.Label(stay_frame, text="STAYED", bg=PANEL, fg=GRAY,
                 font=("Courier New", 9, "bold")).pack()
        self.stay_stat = tk.Label(stay_frame, text="0 / 0  (0%)",
                                  bg=PANEL, fg=GREEN, font=("Courier New", 13, "bold"))
        self.stay_stat.pack()

        switch_frame = tk.Frame(stats_row, bg=PANEL, padx=18, pady=10)
        switch_frame.grid(row=0, column=1, padx=14)
        tk.Label(switch_frame, text="SWITCHED", bg=PANEL, fg=GRAY,
                 font=("Courier New", 9, "bold")).pack()
        self.switch_stat = tk.Label(switch_frame, text="0 / 0  (0%)",
                                    bg=PANEL, fg=ACCENT, font=("Courier New", 13, "bold"))
        self.switch_stat.pack()

        # Bottom buttons
        btn_row = tk.Frame(self.root, bg=BG)
        btn_row.pack(pady=(14, 20))

        self.reset_btn = tk.Button(btn_row, text="New Game",
                                   command=self._reset_game,
                                   bg=GRAY, fg=WHITE,
                                   font=("Courier New", 10, "bold"),
                                   relief="flat", padx=18, pady=7,
                                   activebackground=WHITE, activeforeground=BG,
                                   cursor="hand2")
        self.reset_btn.grid(row=0, column=0, padx=8)

        self.clear_btn = tk.Button(btn_row, text="Clear Stats",
                                   command=self._clear_stats,
                                   bg=PANEL, fg=GRAY,
                                   font=("Courier New", 10),
                                   relief="flat", padx=18, pady=7,
                                   activebackground=GRAY, activeforeground=WHITE,
                                   cursor="hand2")
        self.clear_btn.grid(row=0, column=1, padx=8)

    def _draw_door_closed(self, idx, highlight=False, selected=False):
        c = self.door_canvases[idx]
        c.delete("all")
        border = ACCENT if selected else (GOLD if highlight else DOOR_CLOSED)
        c.config(highlightbackground=border)

        # Door body
        c.create_rectangle(10, 10, 120, 170, fill=DOOR_CLOSED, outline=border, width=2)
        # Door panels
        c.create_rectangle(18, 18, 65, 90, fill="#0a2744", outline=border, width=1)
        c.create_rectangle(68, 18, 115, 90, fill="#0a2744", outline=border, width=1)
        c.create_rectangle(18, 98, 115, 162, fill="#0a2744", outline=border, width=1)
        # Knob
        c.create_oval(88, 88, 96, 96, fill=GOLD, outline=GOLD)

        num_color = ACCENT if selected else GOLD
        c.create_text(65, 140, text=f"{idx+1}", fill=num_color,
                      font=("Courier New", 28, "bold"))

    def _draw_door_open_goat(self, idx):
        c = self.door_canvases[idx]
        c.delete("all")
        c.config(highlightbackground=DOOR_OPEN_GOAT)
        c.create_rectangle(10, 10, 120, 170, fill=DOOR_OPEN_GOAT, outline=DOOR_OPEN_GOAT, width=2)
        # Goat drawing (simple shapes)
        # body
        c.create_oval(35, 90, 95, 140, fill="#8B7355", outline="#6B5335", width=1)
        # head
        c.create_oval(70, 65, 105, 100, fill="#8B7355", outline="#6B5335", width=1)
        # legs
        c.create_line(45, 138, 40, 165, fill="#6B5335", width=4)
        c.create_line(60, 140, 58, 165, fill="#6B5335", width=4)
        c.create_line(75, 140, 73, 165, fill="#6B5335", width=4)
        c.create_line(88, 138, 86, 165, fill="#6B5335", width=4)
        # eye
        c.create_oval(90, 73, 95, 78, fill="black")
        # ear
        c.create_polygon(73, 68, 68, 52, 78, 58, fill="#8B7355", outline="#6B5335")
        # horns
        c.create_line(78, 66, 73, 50, fill="#A0896B", width=2)
        c.create_line(87, 67, 93, 52, fill="#A0896B", width=2)
        # label
        c.create_text(65, 22, text="GOAT", fill=WHITE, font=("Courier New", 9, "bold"))

    def _draw_door_open_car(self, idx):
        c = self.door_canvases[idx]
        c.delete("all")
        c.config(highlightbackground=DOOR_OPEN_CAR)
        c.create_rectangle(10, 10, 120, 170, fill="#1a1a1a", outline=DOOR_OPEN_CAR, width=2)
        # Car body bottom
        c.create_rectangle(20, 110, 110, 150, fill=ACCENT, outline="#c73348", width=1)
        # Car roof
        c.create_polygon(35, 110, 45, 80, 85, 80, 95, 110, fill=ACCENT, outline="#c73348")
        # Windows
        c.create_rectangle(48, 84, 72, 108, fill="#add8e6", outline="#87ceeb", width=1)
        c.create_rectangle(74, 84, 90, 108, fill="#add8e6", outline="#87ceeb", width=1)
        # Wheels
        c.create_oval(22, 140, 50, 165, fill="#222", outline=GRAY, width=2)
        c.create_oval(78, 140, 106, 165, fill="#222", outline=GRAY, width=2)
        c.create_oval(29, 147, 43, 158, fill="#555", outline=GRAY)
        c.create_oval(85, 147, 99, 158, fill="#555", outline=GRAY)
        # Headlight
        c.create_rectangle(100, 120, 112, 130, fill=GOLD, outline=GOLD)
        # label
        c.create_text(65, 22, text="CAR !", fill=GOLD, font=("Courier New", 9, "bold"))

    def _reset_game(self):
        self.car_door = random.randint(0, 2)
        self.player_choice = None
        self.revealed_door = None
        self.game_phase = "choose"

        for i in range(3):
            self._draw_door_closed(i)
            self.door_buttons[i].config(text="Pick Door", state="normal",
                                        bg=DOOR_CLOSED, fg=WHITE)
            self.door_labels[i].config(fg=WHITE)

        self.msg_var.set("Pick a door! The car is hidden behind one of them.")

    def _on_door_click(self, idx):
        if self.game_phase == "choose":
            self._player_picks(idx)
        elif self.game_phase == "revealed":
            self._player_final_choice(idx)

    def _player_picks(self, idx):
        self.player_choice = idx
        self.game_phase = "revealed"

        # Highlight chosen door
        for i in range(3):
            self._draw_door_closed(i, selected=(i == idx))

        # Monty reveals a goat door (not car, not player's choice)
        choices = [i for i in range(3) if i != idx and i != self.car_door]
        self.revealed_door = random.choice(choices)
        self._draw_door_open_goat(self.revealed_door)

        # Update buttons
        for i in range(3):
            if i == self.revealed_door:
                self.door_buttons[i].config(text="(Goat)", state="disabled",
                                            bg="#1a3a1a", fg=GRAY)
                self.door_labels[i].config(fg=GRAY)
            elif i == idx:
                self.door_buttons[i].config(text="Stay Here", bg=ACCENT, fg=WHITE)
            else:
                self.door_buttons[i].config(text="Switch!", bg=GREEN, fg=WHITE)

        self.msg_var.set(
            f"Monty opens Door {self.revealed_door + 1} -- it's a goat!\n"
            f"You picked Door {idx + 1}. Stay or switch?"
        )

    def _player_final_choice(self, idx):
        if idx == self.revealed_door:
            return

        self.game_phase = "done"
        switched = (idx != self.player_choice)
        won = (idx == self.car_door)

        # Reveal all doors
        for i in range(3):
            if i == self.car_door:
                self._draw_door_open_car(i)
            else:
                if i != self.revealed_door:
                    self._draw_door_open_goat(i)
            self.door_buttons[i].config(state="disabled")

        # Update stats
        if switched:
            self.stats["switch_total"] += 1
            if won:
                self.stats["switch_wins"] += 1
        else:
            self.stats["stay_total"] += 1
            if won:
                self.stats["stay_wins"] += 1

        self._update_stats_display()

        action = "SWITCHED" if switched else "STAYED"
        result = "WIN! You got the car!" if won else "LOSS. It was a goat."

        color = GREEN if won else ACCENT
        self.msg_label.config(fg=color)
        self.msg_var.set(f"You {action} to Door {idx + 1}  -->  {result}")

    def _update_stats_display(self):
        sw = self.stats["stay_wins"]
        st = self.stats["stay_total"]
        sp = int(100 * sw / st) if st else 0
        self.stay_stat.config(text=f"{sw} / {st}  ({sp}%)")

        sww = self.stats["switch_wins"]
        swt = self.stats["switch_total"]
        swp = int(100 * sww / swt) if swt else 0
        self.switch_stat.config(text=f"{sww} / {swt}  ({swp}%)")

    def _clear_stats(self):
        self.stats = {"stay_wins": 0, "stay_total": 0, "switch_wins": 0, "switch_total": 0}
        self.stay_stat.config(text="0 / 0  (0%)")
        self.switch_stat.config(text="0 / 0  (0%)")


if __name__ == "__main__":
    root = tk.Tk()
    app = MontyHallApp(root)
    root.mainloop()
