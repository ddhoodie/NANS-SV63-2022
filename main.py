import json
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QLineEdit, QCheckBox, QPushButton, \
    QFormLayout, QDesktopWidget, QHBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtCore import QTimer, Qt

from model import predict, RegressionModel, test_model


class DisclaimerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Disclaimer')

        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - 400) // 2
        y = (screen_geometry.height() - 200) // 2
        self.setGeometry(x, y, 400, 400)

        layout = QVBoxLayout()

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        disclaimer_label = QLabel("Disclaimer: This prediction app uses a model that relies solely on steam "
                                  "statistics. \n Predictions made by this app is given in steam player counts. \n "
                                  "This app is still in development.")
        layout.addWidget(disclaimer_label, alignment=Qt.AlignCenter)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        button_layout = QHBoxLayout()

        button_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        continue_button = QPushButton("Continue")
        continue_button.clicked.connect(self.start_main_menu)
        button_layout.addWidget(continue_button)

        layout.addLayout(button_layout)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.start_main_menu)
        self.timer.start(10000)

        # Set background color to blue and text color to white
        self.setStyleSheet("background-color: #252F9C; color: #CFE5FF;")

    def start_main_menu(self):
        self.main_menu = MainMenuWindow()
        self.main_menu.show()
        self.timer.stop()
        self.close()


class MainMenuWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Game Predictor')

        # Calculate the position to center the window
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - 400) // 2  # Center horizontally
        y = (screen_geometry.height() - 200) // 2  # Center vertically
        self.setGeometry(x, y, 800, 600)

        layout = QVBoxLayout()
        welcome_label = QLabel("<span style='font-size: 12pt;'><b>Welcome to Game Predictor!</b></span>")
        layout.addWidget(welcome_label)

        form_layout = QFormLayout()

        self.name_edit = QLineEdit()
        form_layout.addRow("Name:", self.name_edit)

        self.year_of_release_edit = QLineEdit()
        form_layout.addRow("Year of Release:", self.year_of_release_edit)

        self.company_budget_edit = QLineEdit()
        form_layout.addRow("Company Budget:", self.company_budget_edit)

        self.trailer_views_edit = QLineEdit()
        form_layout.addRow("Trailer Views:", self.trailer_views_edit)

        self.multiplayer_checkbox = QCheckBox()
        form_layout.addRow("Multiplayer:", self.multiplayer_checkbox)

        self.platform_playstation_checkbox = QCheckBox()
        self.platform_playstation_checkbox.setStyleSheet("color: #CFE5FF;")
        form_layout.addRow("PlayStation:", self.platform_playstation_checkbox)

        self.platform_xbox_checkbox = QCheckBox()
        self.platform_xbox_checkbox.setStyleSheet("color: #CFE5FF;")
        form_layout.addRow("Xbox:", self.platform_xbox_checkbox)

        self.action_checkbox = QCheckBox()
        self.action_checkbox.setStyleSheet("color: #CFE5FF;")
        form_layout.addRow("Action:", self.action_checkbox)

        self.adventure_checkbox = QCheckBox()
        self.adventure_checkbox.setStyleSheet("color: #CFE5FF;")
        form_layout.addRow("Adventure:", self.adventure_checkbox)

        self.rpg_checkbox = QCheckBox()
        self.rpg_checkbox.setStyleSheet("color: #CFE5FF;")
        form_layout.addRow("RPG:", self.rpg_checkbox)

        self.simulation_checkbox = QCheckBox()
        self.simulation_checkbox.setStyleSheet("color: #CFE5FF;")
        form_layout.addRow("Simulation:", self.simulation_checkbox)

        self.sports_checkbox = QCheckBox()
        self.sports_checkbox.setStyleSheet("color: #CFE5FF;")
        form_layout.addRow("Sports:", self.sports_checkbox)

        self.puzzle_checkbox = QCheckBox()
        self.puzzle_checkbox.setStyleSheet("color: #CFE5FF;")
        form_layout.addRow("Puzzle:", self.puzzle_checkbox)

        self.horror_checkbox = QCheckBox()
        self.horror_checkbox.setStyleSheet("color: #CFE5FF;")
        form_layout.addRow("Horror:", self.horror_checkbox)

        self.survival_checkbox = QCheckBox()
        self.survival_checkbox.setStyleSheet("color: #CFE5FF;")
        form_layout.addRow("Survival:", self.survival_checkbox)

        self.fps_checkbox = QCheckBox()
        self.fps_checkbox.setStyleSheet("color: #CFE5FF;")
        form_layout.addRow("FPS:", self.fps_checkbox)

        self.mmo_checkbox = QCheckBox()
        self.mmo_checkbox.setStyleSheet("color: #CFE5FF;")
        form_layout.addRow("MMO:", self.mmo_checkbox)

        self.open_world_checkbox = QCheckBox()
        self.open_world_checkbox.setStyleSheet("color: #CFE5FF;")
        form_layout.addRow("Open World:", self.open_world_checkbox)

        self.story_mode_checkbox = QCheckBox()
        self.story_mode_checkbox.setStyleSheet("color: #CFE5FF;")
        form_layout.addRow("Story Mode:", self.story_mode_checkbox)

        self.strategy_checkbox = QCheckBox()
        self.strategy_checkbox.setStyleSheet("color: #CFE5FF;")
        form_layout.addRow("Strategy:", self.strategy_checkbox)

        layout.addLayout(form_layout)
        self.setLayout(layout)

        self.submit_button = QPushButton("Submit")
        layout.addWidget(self.submit_button)

        self.GTA6 = QPushButton("GTA 6")
        layout.addWidget(self.GTA6)

        central_widget = QWidget()
        central_widget.setLayout(layout)  # Set the layout to the central widget
        self.setCentralWidget(central_widget)  # Set the central widget

        self.submit_button.clicked.connect(self.submit_game_details)
        self.GTA6.clicked.connect(self.submit_GTA6)

        self.result_label = QLabel("<span style='font-size: 14pt;'><b>Prediction Result:</b></span>")
        layout.addWidget(self.result_label)

        self.show()  # Ensure the window is shown

        # Set background color to blue and text color to white
        self.setStyleSheet("background-color: #252F9C; color: #CFE5FF;")

    def submit_GTA6(self):
        value = predict(1, "data/gta6.json")
        self.result_label.setText(f"Prediction Result: {value}")

    def submit_game_details(self):
        data = {
            "name": self.name_edit.text(),
            "players_on_launch": 0,
            "players_after_1year": 0,
            "year_of_release": int(self.year_of_release_edit.text()),
            "company_budget": int(self.company_budget_edit.text()),
            "trailer_views": int(self.trailer_views_edit.text()),
            "multiplayer": self.multiplayer_checkbox.isChecked(),
            "platform_availability": {
                "PC": True,
                "PLAYSTATION": self.platform_playstation_checkbox.isChecked(),
                "XBOX": self.platform_xbox_checkbox.isChecked()
            },
            "genre": {
                "action": self.action_checkbox.isChecked(),
                "adventure": self.adventure_checkbox.isChecked(),
                "rpg": self.rpg_checkbox.isChecked(),
                "simulation": self.simulation_checkbox.isChecked(),
                "sports": self.sports_checkbox.isChecked(),
                "puzzle": self.puzzle_checkbox.isChecked(),
                "horror": self.horror_checkbox.isChecked(),
                "survival": self.survival_checkbox.isChecked(),
                "indie": False,
                "fps": self.fps_checkbox.isChecked(),
                "mmo": self.mmo_checkbox.isChecked(),
                "open_world": self.open_world_checkbox.isChecked(),
                "story_mode": self.story_mode_checkbox.isChecked(),
                "strategy": self.strategy_checkbox.isChecked()
            }
        }
        filename = "data/" + data["name"].replace(" ", "_") + ".json"
        with open(filename, 'w') as f:
            json.dump(data, f)

        value = predict(1, filename)
        self.result_label.setText(f"<span style='font-weight: bold; font-size: 14pt;'>"
                                  f"Prediction Result:</span> {value}")


if __name__ == "__main__":
    # test_model(1, RegressionModel.OLS)
    app = QApplication(sys.argv)
    game_prediction_window = DisclaimerWindow()
    game_prediction_window.show()
    sys.exit(app.exec_())
