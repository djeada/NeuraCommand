from app.menu.menu_interface import AbstractMenu, ChoiceBasedMenu
from app.menu.model_managment_menu.io_menu.save_pickle_menu import SavePickleMenu
from app.menu.model_managment_menu.use_model_menu.make_prediction_menu import (
    MakePredictionMenu,
)
from app.menu.model_managment_menu.use_model_menu.train_network_menu import (
    TrainNetworkMenu,
)
from neura_command.deprecated.feed_forward_network.feed_forward_network import (
    FeedForwardNetwork,
)


class UseModelMenu(ChoiceBasedMenu):
    def __init__(self, parent_menu: AbstractMenu, model: FeedForwardNetwork):
        menu_options = {
            "1": {"text": "Make Prediction", "action": self.enter_prediction_menu},
            "2": {"text": "Train Network", "action": self.enter_training_menu},
            "3": {"text": "Export Model as Pickle", "action": self.enter_export_menu},
            "back": {"text": "Return to Main Menu", "action": self.deactivate},
        }
        super().__init__(
            parent_menu=parent_menu,
            name="Model Management Menu",
            menu_options=menu_options,
        )
        self.model = model

    def enter_prediction_menu(self):
        prediction_menu = MakePredictionMenu(parent_menu=self, network=self.model)
        prediction_menu.run()

    def enter_training_menu(self):
        training_menu = TrainNetworkMenu(parent_menu=self, network=self.model)
        training_menu.run()

    def enter_export_menu(self):
        export_menu = SavePickleMenu(parent_menu=self, network=self.model)
        export_menu.run()
