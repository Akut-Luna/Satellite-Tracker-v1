import sys
from PySide6.QtWidgets import QApplication
from Main.UI.UI import SatelliteTrackerApp  # Import the SatelliteTrackerApp class

app = QApplication(sys.argv)
window = SatelliteTrackerApp()
window.show()

sys.exit(app.exec())
