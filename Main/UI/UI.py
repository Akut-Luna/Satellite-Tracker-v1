print('Loading...')
import os
import sys
import json
import socket
import spiceypy
import requests
import traceback
import threading
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.geodesic as geodesic
import matplotlib.image as mpimg
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from astropy import units as astropy_units
from astropy.time import Time as astropy_time
from astropy.coordinates import EarthLocation as astropy_EarthLocation
from datetime import datetime, timezone, timedelta
from skyfield.api import wgs84, EarthSatellite, load
from skyfield.positionlib import Barycentric, position_of_radec
from skyfield.iokit import parse_tle_file
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox, 
    QDateTimeEdit, QRadioButton, QButtonGroup, QFileDialog,
    QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QStackedWidget, QFrame
)
from PySide6.QtCore import QDateTime, Qt, QTimer, QTimeZone
from PySide6.QtGui import QIcon
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas # must be imported after PySide

class NexPassVisualisationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Next Pass Visualisation')
        self.setWindowIcon(QIcon(os.path.join('Main', 'Images', 'satellite_icon_white.svg')))
        layout = QVBoxLayout(self)
        
        self.fig, self.ax = plt.subplots()
        canvas = FigureCanvas(self.fig)
        layout.addWidget(canvas)

        self.ax.set_aspect('equal')
        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)

        # draw concentric circles for elevation
        for r in [1, 2/3, 1/3, 0]:  # 0°, 30°, 60° elevation rings
            circle = plt.Circle((0, 0), r, fill=False, color='gray')
            self.ax.text(0.1, r, f'{int((1-r)*90)}°', ha='center', va='bottom', fontsize=12, color='gray')
            self.ax.add_patch(circle)

        self.ax.hlines([0], [-1.05], [1.05], color='gray')
        self.ax.vlines([0], [-1.05], [1.05], color='gray')

        # Cardinal labels
        self.ax.text(0, 1.1, 'N', ha='center', va='bottom', fontsize=10)
        self.ax.text(1.1, 0, 'E', ha='left', va='center', fontsize=10)
        self.ax.text(0, -1.1, 'S', ha='center', va='top', fontsize=10)
        self.ax.text(-1.1, 0, 'W', ha='right', va='center', fontsize=10)

        self.ax.axis('off')

    def az_el_to_xy(self, az_deg, el_deg):
        r = (90 - el_deg) / 90  # scale to [0,1]
        theta = -np.radians(az_deg) + np.pi/2 # N is up and azimuth increases cockwise
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y
    
    def label_plot(self, text, x, y):
        if x > 0: # right
            self.ax.text(x+0.05, y, text, ha='left', va='center', fontsize=10)
        if x < 0: # left
            self.ax.text(x-0.05, y, text, ha='right', va='center', fontsize=10)

    def plot(self, data, start_time, end_time, UTC):
        '''
        Parameters:
            data (list): shape (N,2) az, el values for the next path
            start_time (datetime): time of AOS 
            end_time (datetime): time of LOS
            UTC (bool): flag that shows if time is in UTC or Local Time
        '''
        az = data[:,0]
        el = data[:,1]

        x, y = self.az_el_to_xy(az, el)
        self.ax.plot(x,y)

        # Labels
        if UTC:
            tz = 'UTC'
        else:
            tz = 'Local Time'
        
        start_time_str = str(start_time.time()).split('.')[0]
        end_time_str = str(end_time.time()).split('.')[0]
            
        self.label_plot(f'AOS ({tz})\n{start_time_str}', x[0], y[0])
        self.label_plot(f'LOS ({tz})\n{end_time_str}', x[-1], y[-1])

class SatelliteTrackerApp(QMainWindow):
    def __init__(self):
        '''
        This function initializes the class and set up all 'global' variables.
        '''
        super().__init__()
        self.setWindowTitle('Satellite Tracker')
        self.setWindowIcon(QIcon(os.path.join('Main', 'Images', 'satellite_icon_white.svg')))
        self.setGeometry(100, 100, 1200, 800) # set inital pos and size

        # Initialize 'global' variables
        load_dotenv(os.path.join('Main', 'config', 'config.env'))
        self.antenna_latitude = float(os.getenv('LATITUDE'))
        self.antenna_longitude = float(os.getenv('LONGITUDE'))
        self.antenna_altitude = float(os.getenv('ALTITUDE'))
        self.min_angle_change_before_update = float(os.getenv('MIN_ANGLE_CHANGE_BEFORE_UPDATE'))
        self.local_tz = os.getenv('LOCAL_TZ') # local time zone
        self.display_light_time_correction_option = os.getenv('DISPLAY_LIGHT_TIME_CORRECTION_OPTION').upper() == 'TRUE'

        self.skyfield_antenna_pos = wgs84.latlon(self.antenna_latitude, self.antenna_longitude, self.antenna_altitude)
        self.skyfield_ts = load.timescale() # used to create skyfield time objects
        
        planet_ephemeris_path = os.path.join('Main', 'data', 'Ephemeris', 'de421.bsp')
        self.planet_ephemeris = load(planet_ephemeris_path)
        
        self.tracking = False

        # data
        self.satellite_list = self.get_satellites_from_file()
        self.satellite_metadata = self.load_satellite_metadata()
        self.all_CelesTrak_satellites_df = None # gets loaded by load_all_satellite_data()
        self.omm_df = None                      # gets loaded by browse_gp_file()
        self.tle_by_name = None                 # gets loaded by browse_gp_file()
        self.tle_by_norad = None                # gets loaded by browse_gp_file()
        self.tle_by_intl = None                 # gets loaded by browse_gp_file()
        self.gp_file_satellite = None           # gets set by tracking_mode_TLE_OMM()

        # SPICE
        self.spice_kernels_loaded = False # gets set by browse_spice_file()

        # timer for continuously updating
        self.update_frequency = 500 # ms
        self.update_continuously_timer = QTimer(self)
        self.update_continuously_timer.timeout.connect(self.update_continuously)

        # Motors
        self.motor_controller_IP = os.getenv('IP_ADRESS')
        self.motor_controller_port = int(os.getenv('PORT'))
        self.socket = None
        self.last_time_motor_got_updated = None

        # Map
        map_path = os.path.join('Main', 'Images', 'nasa-topo_1024.jpg')
        self.earth_img = mpimg.imread(map_path)
        self.min_before_recalculate_flight_path = int(os.getenv('MIN_BEFORE_RECALCULATING_FLIGHT_PATH'))
        self.flight_path_steps = int(os.getenv('FLIGHT_PATH_STEPS'))
        self.last_time_flight_path_got_calculated = None
        self.flight_path = None

        # UI
        self.setup_ui()

        # --- processes that need to happen after the UI is set up, because they log to UI console ---

        # load local satellite data
        self.load_all_satellite_data()
        
        # check if motor controller is available
        self.motor_controller_establish_connection()
        
        # start continuous update
        self.update_continuously_timer.start(self.update_frequency)

    # UI setup ------------------------------------------------------------------------------------ 
    def set_style(self):
        '''
        Set font size and maximum size of UI elements
        '''

        app = QApplication.instance()
        app.setStyleSheet(f'''
            * {{
                font-size: 15px;
            }}
        ''')

        # Find Passes =============================================================================
        self.find_passes_group.setMaximumHeight(210)

        # Tracking Options ========================================================================
        self.tracking_modes_group.setMaximumHeight(210)

        # List ------------------------------------------------------------------------------------
        self.tracking_mode_list_dropdown.setMaxVisibleItems(20)

        # RA/DEC ----------------------------------------------------------------------------------
        self.ra_dec_widget.setMaximumWidth(300)

        # TLE/OMM File ----------------------------------------------------------------------------
        # self.gp_file_add_to_list_btn.setMaximumWidth(100)

        # SPICE -----------------------------------------------------------------------------------

        # AZ/EL -----------------------------------------------------------------------------------
        self.az_el_widget.setMaximumWidth(300)

        # Antenna =================================================================================
        self.antenna_group.setMaximumSize(450, 194)
        
        # Azimuth ---------------------------------------------------------------------------------
        self.azimuth_label.setMaximumWidth(80)
        self.current_azimuth.setMaximumWidth(60)
        self.target_azimuth.setMaximumWidth(60)
        self.azimuth_offset.setMaximumWidth(85)
        self.azimuth_offset_reset_btn.setMaximumWidth(80)
        
        # Elevation -------------------------------------------------------------------------------
        self.elevation_label.setMaximumWidth(80)
        self.current_elevation.setMaximumWidth(60)
        self.target_elevation.setMaximumWidth(60)
        self.elevation_offset.setMaximumWidth(85)
        self.elevation_offset_reset_btn.setMaximumWidth(80)

        # Doppler Shift ---------------------------------------------------------------------------
        self.doppler_initial_freq.setMaximumWidth(150)
        self.doppler_shifted_freq.setMaximumWidth(150)
        self.doppler_shift_label.setMaximumWidth(100)

        # Data ====================================================================================
        self.data_group.setMaximumSize(300, 194)

        # Tracking ================================================================================
        self.tracking_group.setMaximumHeight(194)

        self.tracking_btn.setMaximumWidth(170)
        self.tracking_layout.setAlignment(Qt.AlignCenter)

        # Map =====================================================================================

        # Console =================================================================================

    def setup_find_passes_widget(self):
        '''
        Sets up the UI element 'Find Passes'
        '''
        self.find_passes_group = QGroupBox('Find Passes')
        find_passes_layout = QGridLayout(self.find_passes_group)

        # Radio buttons for UTC / Local Time
        self.time_zone_group = QButtonGroup(self)
        self.utc_radio_button = QRadioButton('UTC')
        self.utc_radio_button.setChecked(True)  # Default to UTC
        self.local_time_radio_button = QRadioButton('Local Time')
        self.time_zone_group.addButton(self.utc_radio_button)
        self.time_zone_group.addButton(self.local_time_radio_button)

        find_passes_layout.addWidget(self.utc_radio_button, 0, 0)
        find_passes_layout.addWidget(self.local_time_radio_button, 0, 1)
        
        # Connect the radio button change signal to a function
        self.time_zone_group.buttonToggled.connect(self.UTC_local_time_button_func)

        # Start time
        find_passes_layout.addWidget(QLabel('Start time:'), 1, 0)
        self.start_time_input = QDateTimeEdit()
        self.start_time_input.setDateTime(QDateTime.currentDateTime())
        self.start_time_input.setTimeZone(QTimeZone(b'UTC'))
        self.start_time_input.setDisplayFormat('hh:mm dd.MM.yyyy')
        find_passes_layout.addWidget(self.start_time_input, 1, 1)
        
        # End time
        find_passes_layout.addWidget(QLabel('End time:'), 2, 0)
        self.end_time_input = QDateTimeEdit()
        self.end_time_input.setDateTime(QDateTime.currentDateTime().addDays(1))
        self.end_time_input.setTimeZone(QTimeZone(b'UTC'))
        self.end_time_input.setDisplayFormat('hh:mm dd.MM.yyyy')
        find_passes_layout.addWidget(self.end_time_input, 2, 1)
        
        # Min elevation
        find_passes_layout.addWidget(QLabel('Min elevation:'), 3, 0)
        self.min_elevation_input = QSpinBox()
        self.min_elevation_input.setRange(0, 90)
        self.min_elevation_input.setValue(0)
        self.min_elevation_input.setSuffix('°')
        find_passes_layout.addWidget(self.min_elevation_input, 3, 1)
        
        # Find passes button
        self.find_passes_btn = QPushButton('Find Passes')
        self.find_passes_btn.clicked.connect(self.find_passes)
        find_passes_layout.addWidget(self.find_passes_btn, 4, 0, 1, 2)

        # Next Pass Visualisation Button
        self.next_pass_visualisation_btn = QPushButton('Visualise Next Pass')
        self.next_pass_visualisation_btn.clicked.connect(self.visualise_next_pass)
        find_passes_layout.addWidget(self.next_pass_visualisation_btn, 5, 0, 1, 2)

        self.top_layout.addWidget(self.find_passes_group)
    
    def setup_tracking_modes_widget(self):
        '''
        Sets up the UI element 'Tracking Modes' (former 'Tracking Options')
        '''
        self.tracking_modes_group = QGroupBox('Tracking Modes')
        tracking_modes_layout = QVBoxLayout(self.tracking_modes_group)
        
        # tracking option selection
        self.tracking_mode_combo = QComboBox()
        self.tracking_mode_combo.addItems(['List', 'RA/DEC', 'TLE/OMM File', 'SPICE', 'AZ/EL'])
        self.tracking_mode_combo.currentIndexChanged.connect(self.on_tracking_mode_changed)
        tracking_modes_layout.addWidget(self.tracking_mode_combo)
        
        # Stacked widget to switch between tracking options input types
        self.tracking_mode_stack = QStackedWidget()
        
        # 0. List widget --------------------------------------------------------------------------
        self.list_widget = QWidget()
        list_layout = QVBoxLayout(self.list_widget)
        self.tracking_mode_list_dropdown = QComboBox()
        self.tracking_mode_list_dropdown.addItems(self.get_satellite_names_from_file())
        self.tracking_mode_list_dropdown.currentIndexChanged.connect(self.on_tracking_mode_list_dropdown_changed)
        list_layout.addWidget(self.tracking_mode_list_dropdown)
        self.tracking_mode_stack.addWidget(self.list_widget)
        
        # 1. RA/DEC widget ------------------------------------------------------------------------
        self.ra_dec_widget = QWidget()
        ra_dec_layout = QGridLayout(self.ra_dec_widget)

        
        ra_dec_layout.addWidget(QLabel('RA [h]:'), 0, 0)
        self.ra_input = QLineEdit()
        ra_dec_layout.addWidget(self.ra_input, 0, 1)

        ra_dec_layout.addWidget(QLabel('DEC [°]:'), 1, 0)
        self.dec_input = QLineEdit()
        ra_dec_layout.addWidget(self.dec_input, 1, 1)
        self.tracking_mode_stack.addWidget(self.ra_dec_widget)
        
        # 2. TLE/OMM File widget ------------------------------------------------------------------
        self.gp_file_widget = QWidget()
        gp_file_layout = QVBoxLayout(self.gp_file_widget)

        # top -----------------------------------
        gp_file_top_layout = QHBoxLayout()

        gp_file_top_layout.addWidget(QLabel('TLE/OMM file:'))
        self.gp_file_input = QLineEdit()
        self.gp_file_input.setReadOnly(True)
        gp_file_top_layout.addWidget(self.gp_file_input)

        # browse button
        self.gp_file_browse_btn = QPushButton('Browse')
        self.gp_file_browse_btn.clicked.connect(self.browse_gp_file)
        gp_file_top_layout.addWidget(self.gp_file_browse_btn)

        gp_file_layout.addLayout(gp_file_top_layout)

        # middle --------------------------------
        gp_file_middle_layout = QGridLayout()

        # satellite name
        gp_file_middle_layout.addWidget(QLabel('Satellite Name'), 0, 0)
        self.gp_file_satellite_name = QLineEdit()
        gp_file_middle_layout.addWidget(self.gp_file_satellite_name, 1, 0)

        # Int'l ID
        gp_file_middle_layout.addWidget(QLabel("Int'l ID"), 0, 1)
        self.gp_file_intl_id = QLineEdit()
        gp_file_middle_layout.addWidget(self.gp_file_intl_id, 1, 1)

        # NORAD
        gp_file_middle_layout.addWidget(QLabel('NORAD ID'), 0, 2)
        self.gp_file_norad_id = QLineEdit()
        gp_file_middle_layout.addWidget(self.gp_file_norad_id, 1, 2)

        gp_file_layout.addLayout(gp_file_middle_layout)

        # bottom --------------------------------
        gp_file_bottom_layout = QHBoxLayout()

        # add to list button
        self.gp_file_add_to_list_btn = QPushButton('Add to List')
        self.gp_file_add_to_list_btn.clicked.connect(self.add_satellite_to_list)
        gp_file_bottom_layout.addWidget(self.gp_file_add_to_list_btn)
        
        gp_file_layout.addLayout(gp_file_bottom_layout)
        self.tracking_mode_stack.addWidget(self.gp_file_widget)
        
        # 3. SPICE widget -------------------------------------------------------------------------
        self.spice_widget = QWidget()
        spice_layout = QGridLayout(self.spice_widget)

        # (path) input
        spice_layout.addWidget(QLabel('SPICE Meta Kernel:'), 0, 0)
        self.spice_input = QLineEdit()
        self.spice_input.setReadOnly(True)
        spice_layout.addWidget(self.spice_input, 0, 1)

        # button
        self.spice_file_browse_btn = QPushButton('Browse')
        self.spice_file_browse_btn.clicked.connect(self.browse_spice_file)
        spice_layout.addWidget(self.spice_file_browse_btn, 0, 2)

        # Satellite name
        spice_layout.addWidget(QLabel('Satellite Name:'), 1, 0)
        self.spice_name = QLineEdit()
        spice_layout.addWidget(self.spice_name, 1, 1)

        self.tracking_mode_stack.addWidget(self.spice_widget)

        # 4. AZ/EL widget -------------------------------------------------------------------------
        self.az_el_widget = QWidget()
        az_el_layout = QGridLayout(self.az_el_widget)

        az_el_layout.addWidget(QLabel('Azimuth [°]:'), 0, 0)
        self.az_input = QLineEdit()
        az_el_layout.addWidget(self.az_input, 0, 1)

        az_el_layout.addWidget(QLabel('Elevation [°]:'), 1, 0)
        self.el_input = QLineEdit()
        az_el_layout.addWidget(self.el_input, 1, 1)
        self.tracking_mode_stack.addWidget(self.az_el_widget)

        # -----------------------------------------------------------------------------------------

        tracking_modes_layout.addWidget(self.tracking_mode_stack)
        self.top_layout.addWidget(self.tracking_modes_group)

    def setup_antenna_widget(self):
        '''
        Sets up the UI element 'Antenna'
        '''
        # Antenna Group
        self.antenna_group = QGroupBox('Antenna')
        antenna_layout = QVBoxLayout(self.antenna_group)
        
        # ========================================= AZ EL =========================================
        az_el_layout = QGridLayout()

        # Azimuth ---------------------------------------------------------------------------------
        self.azimuth_label = QLabel('Azimuth')
        az_el_layout.addWidget(self.azimuth_label, 1, 0)
        az_el_layout.addWidget(QLabel('Current'), 0, 1)
        self.current_azimuth = QLineEdit('0.0°')
        self.current_azimuth.setReadOnly(True)
        az_el_layout.addWidget(self.current_azimuth, 1, 1)
        
        az_el_layout.addWidget(QLabel('Target'), 0, 2)
        self.target_azimuth = QLineEdit('0.0°')
        self.target_azimuth.setReadOnly(True)
        az_el_layout.addWidget(self.target_azimuth, 1, 2)
        
        az_el_layout.addWidget(QLabel('Offset'), 0, 3)
        self.azimuth_offset = QDoubleSpinBox()
        self.azimuth_offset.setRange(-360, 360)
        self.azimuth_offset.setDecimals(1)
        self.azimuth_offset.setSingleStep(0.1)
        self.azimuth_offset.setValue(0.0)
        self.azimuth_offset.setSuffix('°')
        az_el_layout.addWidget(self.azimuth_offset, 1, 3)

        self.azimuth_offset_reset_btn = QPushButton('reset')
        self.azimuth_offset_reset_btn.clicked.connect(lambda: self.azimuth_offset.setValue(0.0))
        az_el_layout.addWidget(self.azimuth_offset_reset_btn, 1, 4)
        
        # Elevation -------------------------------------------------------------------------------
        self.elevation_label = QLabel('Elevation')
        az_el_layout.addWidget(self.elevation_label, 2, 0)
        self.current_elevation = QLineEdit('0.0°')
        self.current_elevation.setReadOnly(True)
        az_el_layout.addWidget(self.current_elevation, 2, 1)
        
        self.target_elevation = QLineEdit('0.0°')
        self.target_elevation.setReadOnly(True)
        az_el_layout.addWidget(self.target_elevation, 2, 2)
        
        self.elevation_offset = QDoubleSpinBox()
        self.elevation_offset.setRange(-90, 90)
        self.elevation_offset.setDecimals(1)
        self.elevation_offset.setSingleStep(0.1)
        self.elevation_offset.setValue(0.0)
        self.elevation_offset.setSuffix('°')
        az_el_layout.addWidget(self.elevation_offset, 2, 3)

        self.elevation_offset_reset_btn = QPushButton('reset')
        self.elevation_offset_reset_btn.clicked.connect(lambda: self.elevation_offset.setValue(0.0))
        az_el_layout.addWidget(self.elevation_offset_reset_btn, 2, 4)

        # Horizontal line -------------------------------------------------------------------------
        horizontal_line = QFrame()
        horizontal_line.setFrameShape(QFrame.HLine)
        horizontal_line.setFrameShadow(QFrame.Sunken)
        az_el_layout.addWidget(horizontal_line, 3, 0, 1, 5)
        
        antenna_layout.addLayout(az_el_layout)

        # ===================================== Doppler shift =====================================
        doppler_shift_layout = QGridLayout()

        # Doppler shift ---------------------------------------------------------------------------
        self.doppler_shift_label = QLabel('Doppler Shift')
        doppler_shift_layout.addWidget(self.doppler_shift_label, 2, 0)
        doppler_shift_layout.addWidget(QLabel('Emitted freq. [MHz]'), 1, 1)
        self.doppler_initial_freq = QLineEdit()
        self.doppler_initial_freq.setText('0.0')
        doppler_shift_layout.addWidget(self.doppler_initial_freq, 2, 1)

        doppler_shift_layout.addWidget(QLabel('Observed freq. [MHz]'), 1, 2)
        self.doppler_shifted_freq = QLineEdit()
        self.doppler_shifted_freq.setText('0.0')
        self.doppler_shifted_freq.setReadOnly(True)
        doppler_shift_layout.addWidget(self.doppler_shifted_freq, 2, 2)

        antenna_layout.addLayout(doppler_shift_layout)
        self.middle_layout.addWidget(self.antenna_group)

    def setup_data_widget(self):
        '''
        Sets up the UI element 'Data'
        '''
        # Data Group
        self.data_group = QGroupBox('Data')
        data_layout = QGridLayout(self.data_group)

        # UTC ------------------------------------------------------------------------------------
        data_layout.addWidget(QLabel('UTC'), 0, 0)
        self.UTC_text = QDateTimeEdit()
        self.UTC_text.setTimeZone(QTimeZone(b'UTC'))
        self.UTC_text.setDisplayFormat('hh:mm:ss dd.MM.yyyy')
        self.UTC_text.setDateTime(QDateTime.currentDateTimeUtc())
        self.UTC_text.setReadOnly(True)
        data_layout.addWidget(self.UTC_text, 0, 1)

        # Altitude --------------------------------------------------------------------------------
        data_layout.addWidget(QLabel('Altitude'), 1, 0)
        self.altitude_text = QLineEdit()
        self.altitude_text.setText('0 km')
        self.altitude_text.setReadOnly(True)
        data_layout.addWidget(self.altitude_text, 1, 1)

        # Range -----------------------------------------------------------------------------------
        data_layout.addWidget(QLabel('Range'), 2, 0)
        self.range_text = QLineEdit()
        self.range_text.setText('0 km')
        self.range_text.setReadOnly(True)
        data_layout.addWidget(self.range_text, 2, 1)

        # Range Rate ------------------------------------------------------------------------------
        data_layout.addWidget(QLabel('Range Rate'), 3, 0)
        self.range_rate_text = QLineEdit()
        self.range_rate_text.setText('0 km/s')
        self.range_rate_text.setReadOnly(True)
        data_layout.addWidget(self.range_rate_text, 3, 1)
        
        self.middle_layout.addWidget(self.data_group)

    def setup_tracking_widget(self):
        '''
        Sets up the UI element 'Tracking'
        '''
        # Tracking Group
        self.tracking_group = QGroupBox('Tracking')
        self.tracking_layout = QVBoxLayout(self.tracking_group)

        # Start/Stop Tracking button --------------------------------------------------------------
        self.tracking_btn = QPushButton('Start Tracking')
        self.tracking_btn.setCheckable(True)
        self.tracking_btn.toggled.connect(self.toggle_tracking)
        self.tracking_layout.addWidget(self.tracking_btn)

        # Start Tracking at AOS -------------------------------------------------------------------
        self.start_tracking_at_AOS_btn = QRadioButton('Start Tracking at AOS')
        self.tracking_layout.addWidget(self.start_tracking_at_AOS_btn)

        # Light Time Correction button ------------------------------------------------------------
        # Since the Horizon data is already ligth corrected, my light correction is not needed.
        # I'm still leaving this feature in because it might be usefull in the future with data
        # from a different source. In the config file you can set DISPLAY_LIGHT_TIME_CORRECTION_OPTION 
        # to True in order to display this button, that allows the activation of this feature. 
        if self.display_light_time_correction_option:
            self.light_time_correction_btn = QRadioButton('Light Time Correction')
            self.tracking_layout.addWidget(self.light_time_correction_btn)

        self.middle_layout.addWidget(self.tracking_group)

    def setup_ui(self):
        '''
        Sets up the main UI window
        '''
        # Main widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        
        # Top row: find passes and tracking method
        self.top_layout = QHBoxLayout()      
        self.setup_find_passes_widget()
        self.setup_tracking_modes_widget()
        main_layout.addLayout(self.top_layout)
        
        # Middle row: Antenna, Data and Tracking
        self.middle_layout = QHBoxLayout()
        self.setup_antenna_widget()
        self.setup_data_widget()
        self.setup_tracking_widget()
        main_layout.addLayout(self.middle_layout)
        
        # Bottom row: World map and console
        bottom_layout = QHBoxLayout()
        
        # World map
        self.map_projection = ccrs.PlateCarree()
        self.map_figure = Figure(figsize=(8, 4))
        self.map_canvas = FigureCanvas(self.map_figure)
        self.map_ax = self.map_figure.add_subplot(111, projection=self.map_projection)
        self.update_map(None, None, None) # empty map
        
        bottom_layout.addWidget(self.map_canvas)
        
        # Console
        console_group = QGroupBox('Console')
        console_layout = QVBoxLayout(console_group)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        console_layout.addWidget(self.console)
        bottom_layout.addWidget(console_group)
        
        main_layout.addLayout(bottom_layout)

        self.set_style()
        
        # Log initial message
        self.log_message('Satellite Tracker initialized')

        # Set focus to the main window to prevent input widgets from capturing arrow keys
        self.setFocus()

    # time convertions ----------------------------------------------------------------------------
    def local_time_to_UTC(self, datetime):
        '''
        Parameters:
            datetime (datetime): local time

        Returns:
            datetime (datetime): UTC
        '''
        return datetime.replace(tzinfo=ZoneInfo(self.local_tz)).astimezone(ZoneInfo('UTC'))

    def UTC_to_local_time(self, datetime):
        '''
        Parameters:
            datetime (datetime): UTC

        Returns:
            datetime (datetime): local time
        '''

        return datetime.replace(tzinfo=ZoneInfo('UTC')).astimezone(ZoneInfo(self.local_tz))

    def skyfield_time_to_datetime(self, skyfield_time):
        '''
        Prameters:
            skyfield_time (skyfield timescale): skyfield time
        
        Returns:
            datetime (datetime): datetime
        '''
        return datetime.fromisoformat(skyfield_time.utc_iso())

    def datetime_to_skyfield_time(self, datetime):
        '''
        Parameters:
            datetime (datetime): datetime
        
        Returns:
            skyfield_ts (skyfield timescale): skyfield time
        '''
        return self.skyfield_ts.from_datetime(datetime)
    
    def convert_tdb_to_utc(self, jd_tdb, delta_t):
        '''
        Converts TBD Julian Date to UTC

        Parameters:
            jd_tdb (float): Julian Date in Barycentric Dynamical Time (the JPL's T_eph)
            delta_t (float): difference between TBD and UT
        
        Returns:
            jd_utc (float): Julian Date in UTC
        
        NOTE: getting correct DUT1 is not implemented yet, but since DUT1 < 0.9s per definition it can be neglected.
        '''
        jd_ut = jd_tdb - (delta_t / 86400)  # Convert delta-T from seconds to Julian days
        # getting correct DUT1 not implemented yet ------------------------------------------------
        # mjd_ut = int(jd_ut - 2400000.5)     # Convert to Modified Julian Date (MJD)
        # dut1 = self.get_dut1(mjd_ut)        # Get DUT1 from EOP data
        dut1 = 0
        # -----------------------------------------------------------------------------------------
        jd_utc = jd_ut - (dut1 / 86400)     # Apply DUT1 correction
        return jd_utc

    def utc_now(self):
        '''
        Returns:
            datetime (datetime): Current date and time in UTC
        '''
        return datetime.now(timezone.utc)

    # calculations --------------------------------------------------------------------------------
    def get_dut1(self):
        '''
        THIS FUNCTION GETS NOT CALLED AND DOES NOTHING.
        SHOULD THIS EVER GET FIXED YOU NEED TO UPDATE convert_tdb_to_utc() AS WELL.

        Fetch DUT1 (UT1 - UTC) from IERS Bulletin A (EOP data).

        DUT1 is the difference between UT1 (Universal Time) and UTC (Coordinated Universal Time)
        '''
        # for now dut1 = 0. Needs to be fetched automaticaly in the future. 
        # But since per definition DUT < +/-0.9s at all times, DUT = 0 should be close enought
        dut1 = 0 

        return dut1

    def get_locale_sidereal_time(self):
        '''
        Returns:
            LST (astropy Longitude): local sidereal time
        
        alternative manual calculation (http://www.stargazing.net/kepler/altaz.html):
        LST = 100.46 + 0.985647 * d + long + 15*UT

            d    is the days from J2000, including the fraction of a day
            UT   is the universal time in decimal hours
            long is your longitude in decimal degrees, East positive.
            
        Add or subtract multiples of 360 to bring LST in range 0 to 360 degrees.
        '''
        latitude = self.antenna_latitude*astropy_units.deg
        longitude = self.antenna_longitude*astropy_units.deg

        observing_location = astropy_EarthLocation(lat=latitude, lon=longitude)
        observing_time = astropy_time.now()  # Gets the current UTC time dynamically
        LST = observing_time.sidereal_time('mean', longitude=observing_location.lon)
        return LST

    def ra_dec_to_az_el(self, ra_hours, dec_degrees):
        '''
        Parametrs:
            ra_hours (float): Right Ascension in hours
            dec_degrees (float): Declination in degrees

        Returns:
            az (float): Azimuth in degrees
            el (float): Elevation in degrees
        '''

        lat = np.deg2rad(self.antenna_latitude) # rad

        dec = np.deg2rad(dec_degrees) # rad
        ra = np.deg2rad(ra_hours*15)  # rad  

        LST = self.get_locale_sidereal_time() # h
        LST = np.deg2rad(LST.value*15)        # rad
        
        HA = LST - ra # Hour Angle

        # handle bounds https://astrogreg.com/convert_ra_dec_to_alt_az.html
        if HA < 0:
            HA += 2*np.pi
        if HA > np.pi:
            HA -= 2*np.pi
        # -----------------------------------------------------------------
            
        el = np.arcsin(np.sin(dec)*np.sin(lat) + np.cos(dec)*np.cos(lat)*np.cos(HA))
        az = np.arctan2(np.sin(HA), (np.cos(HA)*np.sin(lat) - np.tan(dec)*np.cos(lat)))

        # handle bounds https://astrogreg.com/convert_ra_dec_to_alt_az.html
        az -= np.pi
        if az < 0:
            az += 2*np.pi
        # -----------------------------------------------------------------

        return np.degrees(az), np.degrees(el)

    def doppler_shift(self, f0, range_rate):
        '''
        calculating doppler shifted frequency. 

        Parameters:
            f0 (float): emitted frequency in MHz
            range_rate (float): relative speed in m/s

        Returns:
            f1 (float): observed frequency in MHz
        
        SIGN CONVENTION: 
            if satellite comes closer, frequency need to get up. 
                => 1/(1 - range_rate/c) > 1 
                => 1 > (1 - range_rate/c) 
                => 1 + range_rate/c > 1
                => range_rate > 0
            
            but if the satellite comes closer the range gets smaller => range_rate is neg
                => range_rate needs to change sign
            
            analoge argument when satellite leaves
        '''
        c = 299792458       # m/s
        range_rate *= -1000 # m/s

        f1 = f0 / (1 - range_rate/c) # MHz
        return f1

    def state_vector_ICRF_to_BCRS_position(self, state_vector, t):
        '''
        NOTE: Geometrically the vector in the ICRF and the BCRS position vector are the same. 
        So, what this function does it converting from a 6 dim array into an object that Skyfield understands.

        Parameters:
            state_vector (array or list): 6 dim vector [x, y, z, vx, vy, vz] in km and km/s
            t (skyfield timescale): skyfield time

        Returns:
            Skfield (<Barycentric BCRS position and velocity at date t center=0>): state vector of satellite at time t
        '''
        # convert from km to au
        x, y, z, vx, vy, vz = state_vector
        position_au = np.array([x, y, z]) / 149597870.7  # au
        velocity_au_per_day = np.array([vx, vy, vz]) * 86400 / 149597870.7  # au/day

        satellite_at_t = Barycentric(position_au, velocity_au_per_day, t=t)
        return satellite_at_t
    
    def rotate_by_euler(self, vector, euler_angles):
        '''
        Rotate a vector using Euler angles (ZYX convention).
        
        Parameters:
            vector (numpy.ndarray): 3D Cartesian coordinates [x, y, z]
            euler_angles (tuple): (roll, pitch, yaw) in degrees
        
        Returns:
            rotated_vector (numpy.ndarray): Rotated 3D Cartesian coordinates
        '''
        roll, pitch, yaw = np.radians(euler_angles)
        
        # Rotation matrices
        R_x = np.array([
            [1,      0      ,       0      ],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])
        
        R_y = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [     0        , 1,      0       ],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [     0     ,       0     , 1]
        ])
        
        # Combined rotation matrix (ZYX order)
        R = R_x @ R_y @ R_z
        
        # Apply rotation
        rotated_vector = R @ vector
        
        return rotated_vector

    def az_el_to_cartesian(self, az, el):
        '''
        Convert azimuth and elevation angles to Cartesian coordinates.
        
        Parameters:
            az (float): Azimuth angle in degrees (0° is north, 90° is east)
            el (float): Elevation angle in degrees (0° is horizon, 90° is zenith)
        
        Returns:
            vector (numpy.ndarray): 3D Cartesian coordinates [x, y, z] on a unit sphere
        '''
        # Convert angles from degrees to radians
        az_rad = np.radians(az)
        el_rad = np.radians(el)
        
        # Convert to Cartesian coordinates
        x = np.cos(el_rad) * np.sin(az_rad)
        y = np.cos(el_rad) * np.cos(az_rad)
        z = np.sin(el_rad)
        
        return np.array([x, y, z])
    
    def cartesian_to_az_el(self, vector):
        '''
        Convert Cartesian coordinates to azimuth and elevation angles.
        
        Parameters:
            vector (numpy.ndarray): 3D Cartesian coordinates [x, y, z]
        
        Returns:
            az (float): Azimuth in deg
            el (float): Elevation in deg
        '''
        # Normalize the vector
        vector = vector / np.linalg.norm(vector)
        x, y, z = vector
        
        # Calculate azimuth and elevation
        el = np.degrees(np.arcsin(z))
        az = np.degrees(np.arctan2(x, y))
        
        # Ensure azimuth is in [0, 360) range
        if az < 0:
            az += 360
        
        return az, el

    def correction_matrix(self, az, el, roll, pitch, yaw):
        '''
        Parameters:
            az (float): Azimuth in degree
            el (float): Elevation in degree
            roll (float): roll in degree
            pitch (float): pitch in degree
            yaw (float): yaw in degree
        
        Returns:
            az (float): Azimuth in degree
            el (float): Elevation in degree
        '''
        vector = self.az_el_to_cartesian(az,el)
        vector = self.rotate_by_euler(vector, (roll, pitch, yaw))
        az, el = self.cartesian_to_az_el(vector)
        return az, el

    def calculate_satellite_and_topocentric(self, current_satellite, t):
        '''
        calculates skyfield satellite and topocentric position from Horizons or CelesTrak data
        Parameters:
            current_satellite (dir): data about the satellite from self.satellite_list
            t (skyfield time): time for which the position should be calculated
        
        Returns:
            satellite (Skyfield position): Skyfield satellite position vector at time t
            topocentric (Skyfield position): Skyfield position vector (vector from antenna to satellite)
        '''
        # decide if Horizons or CelesTrak data should be used
        if 'Horizons'  in current_satellite['catalogs']:
            df = current_satellite['df']
            datetime_t = self.skyfield_time_to_datetime(t)

            # Convert time data in df to timezone-aware datetime object if needed
            if isinstance(df['Calendar Date (UTC)'].iloc[0], str):
                df['Calendar Date (UTC)'] = pd.to_datetime(df['Calendar Date (UTC)']).dt.tz_localize('UTC')

            # find two data points closest in time
            closest_rows = df.iloc[(df['Calendar Date (UTC)'] - datetime_t).abs().argsort()[:2]]
            
            # linear interpolation between the two data points ----------------------------
            t1, t2 = closest_rows['Calendar Date (UTC)']
            
            x1, x2 = closest_rows['X']
            y1, y2 = closest_rows['Y']
            z1, z2 = closest_rows['Z']

            vx1, vx2 = closest_rows['VX']
            vy1, vy2 = closest_rows['VY']
            vz1, vz2 = closest_rows['VZ']

            t1 = pd.to_datetime(t1)
            t2 = pd.to_datetime(t2)

            factor = (datetime_t - t1) / (t2 - t1)

            x_now = x1 + factor * (x2 - x1) # km
            y_now = y1 + factor * (y2 - y1) # km
            z_now = z1 + factor * (z2 - z1) # km

            vx_now = vx1 + factor * (vx2 - vx1) # km/s
            vy_now = vy1 + factor * (vy2 - vy1) # km/s
            vz_now = vz1 + factor * (vz2 - vz1) # km/s
            # -----------------------------------------------------------------------------

            state_vector = [x_now, y_now, z_now, vx_now, vy_now, vz_now]
            earth = self.planet_ephemeris['earth'] 
            
            # satellite position vector
            satellite_BCRS_at_t = self.state_vector_ICRF_to_BCRS_position(state_vector, t)

            # Convert to a geocentric position
            satellite_geocentric_at_t = satellite_BCRS_at_t - earth.at(t)

            # relative position object
            topocentric_at_t = satellite_geocentric_at_t - self.skyfield_antenna_pos.at(t)
            satellite_at_t = satellite_geocentric_at_t

        else: # CelesTrack
            # relative position vector
            satellite = current_satellite['EarthSatellite']
            relative_pos = satellite - self.skyfield_antenna_pos 
            
            # relative position object
            topocentric_at_t = relative_pos.at(t)
            satellite_at_t = satellite.at(t) # geocentric
        
        return satellite_at_t, topocentric_at_t

    def should_update_motors(self, current_az, current_el, new_az, new_el):
        '''
        Determines if the newly calculated target azimuth and elevation differ sufficiently from the current position
        such that we need to send the motors new instructions
        Parameters:
            current_az: current azimuth of the antenna. 
            current_el: current elevation of the antenna. 
            new_az: latest value that was calculated for azimuth
            new_el: latest value that was calculated for elevation
        
        Returns:
            (bool): Bool that says if antenna should be moved
        '''
        def az_el_to_vector(azimuth, elevation):
            '''
            NOTE: I'm 99.9% sure that this function can be replaced by self.az_el_to_cartesian
            They use different orientations of the coordiant systems but since we are only interested
            in the angle between 2 vectors that shouldn't matter. But I don't have acesses to the motorcontroller
            at the moment and I don't want to change a tested system with out the ability to test again.


            Convert azimuth and elevation angles (in degrees) to a 3D unit vector.
            Azimuth: angle in the x-y plane from x-axis (0° is +x, 90° is +y)
            Elevation: angle from x-y plane (90° is +z)
            '''
            # Convert degrees to radians
            az_rad = np.radians(azimuth)
            el_rad = np.radians(elevation)
            
            # Calculate the 3D vector components
            x = np.cos(el_rad) * np.cos(az_rad)
            y = np.cos(el_rad) * np.sin(az_rad)
            z = np.sin(el_rad)

            return np.array([x, y, z])
        
        current_vec = az_el_to_vector(current_az, current_el)
        new_vec = az_el_to_vector(new_az, new_el)
                
        # Calculate angle in radians using arccos of the normalized dot product
        dot_product = np.dot(current_vec, new_vec)
        angle_rad = np.arccos(np.clip(dot_product / (np.linalg.norm(current_vec) * np.linalg.norm(new_vec)), -1.0, 1.0))
        
        # Convert to degrees
        angle_deg = np.degrees(angle_rad)
        return angle_deg >= self.min_angle_change_before_update

    def update_data_if_needed(self, current_satellite):
        '''
        Checks how old the current data is and updates if needed.
        
        Parameters:
            current_satellite (dir): data about the satellite from self.satellite_list

        '''
        t = self.utc_now().isoformat()
        t_30_min_later = (datetime.fromisoformat(t) + timedelta(minutes=30)).isoformat()

        # check if Horizons or CelesTrak data is used
        if 'Horizons' in current_satellite['catalogs']:
            satellite_id = current_satellite['catalogs']['Horizons']
            need_to_update = False
            
            # if data does not exist we need to update even during tracking
            if not (str(satellite_id) in self.satellite_metadata['Horizons']):
                need_to_update = True
            else:
                metadata = self.satellite_metadata['Horizons'][f'{satellite_id}']
                
                # if data does not exist or we run out of data we need to update even during tracking
                if metadata['valid until'] == "" or metadata['valid until'] < t:
                    need_to_update = True

                # if we are not tracking and there is less then 30 min of data left we update
                elif not self.tracking and metadata['valid until'] < t_30_min_later:
                    need_to_update = True

            if need_to_update:
                self.log_message(f'Downloading new data for Spacecraft {satellite_id} ...')
                self.query_horizons_api(satellite_id)
                self.load_all_satellite_data(ID=satellite_id)
            
        else:
            last_download = self.satellite_metadata['CelesTrak']['last download']
            time_difference = datetime.fromisoformat(t) - datetime.fromisoformat(last_download)

            # update if data is older then 2h but not while tracking
            if time_difference.total_seconds() > 2 * 3600 and not self.tracking:
                self.log_message('Downloading new CelesTrak data...')
                self.query_celestrak_api()
                self.load_all_satellite_data(celestrak_only=True)
        
    def tracking_mode_List(self, t):
        '''
        Parameters:
            t (skyfield timescale): Skyfield time
        
        Returns:
            az (float): Azimuth in degrees 
            az_rate (float): Azimuth rate in degrees per second
            el (float): Elevation in degrees
            el_rate (float): Elevation rate in degrees per second
            slant_range (float): Distance from antenna to satellite in km
            range_rate (float): Range rate in km/s
            latitude (float): Subpoint latitude in degrees
            longitude (float): Subpoint longitude in degrees
            altitude (float): Altitude of satellite above the ground in km
            f1 (float): Doppler shifted frequency in MHz
        
        NOTE: IF TRACKING IS TURNED OFF, ALL ERROR MESSAGES ARE GETTING IGNORED! 
        The reason for that is that, if the user has not yet finished typing in all necessary information the program would 
        raise lots of errors. So, the idea is that we just display the data that we can calculate with the information
        that we curretly have. However as soon as tracking is turned on we have to assume that the user has entered all 
        necessary information. No we no longer ignore error messages in order to warn the user, if the given information is
        not valid.
        '''
        current_satellite = self.satellite_list[self.tracking_mode_list_dropdown.currentIndex()]
        self.update_data_if_needed(current_satellite)                 

        # get skyfield satellite object and topocentric position object from Horizon or CelesTrak data
        try:
            satellite, topocentric = self.calculate_satellite_and_topocentric(current_satellite, t)
        except Exception as e:
            if self.tracking:
                self.log_message(f'Error calculating satellite position: {str(e)}')
                print(traceback.format_exc())

        el, az, slant_range, el_rate, az_rate, range_rate = topocentric.frame_latlon_and_rates(self.skyfield_antenna_pos)
        
        # light travel time -----------------------------------------------------------------------
        # Since the Horizon data is already ligth corrected, my light correction is not needed.
        # I'm still leaving this feature in because it might be usefull in the future with data
        # from a different source. In the config file you can set DISPLAY_LIGHT_TIME_CORRECTION_OPTION 
        # to True in order to display a button, that allows the activation of this feature. 
        if self.display_light_time_correction_option and self.light_time_correction_btn.isChecked():
            c = 299792458 # m/s                       # if the travel time of the signal gets large
            light_travel_time = slant_range.m/c # s   # we need to take that into account                   
            t -= timedelta(seconds=light_travel_time) # and redo the calculations with an earlier time
            satellite, topocentric = self.calculate_satellite_and_topocentric(current_satellite, t)
            el, az, slant_range, el_rate, az_rate, range_rate = topocentric.frame_latlon_and_rates(self.skyfield_antenna_pos)
        # -----------------------------------------------------------------------------------------

        subpoint = wgs84.subpoint_of(satellite)
        altitude = wgs84.height_of(satellite)
        
        # units -----------------------------------------------------------------------------------
        az = az.degrees
        el = el.degrees
        slant_range = slant_range.km

        az_rate = az_rate.degrees.per_second
        el_rate = el_rate.degrees.per_second
        range_rate = range_rate.km_per_s
        
        latitude = subpoint.latitude.degrees
        longitude = subpoint.longitude.degrees
        altitude = altitude.km

        # doppler shift ---------------------------------------------------------------------------
        # get frequency from config file
        f0 = current_satellite['frequency']
        
        # show initial frequency on UI
        self.doppler_initial_freq.setText(f'{f0:.6f}')

        try:
            f1 = self.doppler_shift(f0, range_rate)
        except Exception as e:
            if self.tracking:
                self.log_message(f'Error calculating doppler shift: {str(e)}')
                print(traceback.format_exc())

        # flight path -----------------------------------------------------------------------------
        now_datetime = self.skyfield_time_to_datetime(t)
        if self.last_time_flight_path_got_calculated is not None:
            delta_t_min = (now_datetime - self.last_time_flight_path_got_calculated).total_seconds() // 60
        else:
            delta_t_min = self.min_before_recalculate_flight_path
            self.last_time_flight_path_got_calculated = now_datetime

        if delta_t_min >= self.min_before_recalculate_flight_path:
            try:
                flight_path = np.zeros((self.flight_path_steps,2))
                for i in range(self.flight_path_steps):
                    t = self.datetime_to_skyfield_time(now_datetime + timedelta(minutes=i))
                    satellite, _ = self.calculate_satellite_and_topocentric(current_satellite, t)
                    subpoint = wgs84.subpoint_of(satellite)
                    flight_path[i][0] = subpoint.latitude.degrees
                    flight_path[i][1] = subpoint.longitude.degrees
                
                self.flight_path = flight_path
                self.last_time_flight_path_got_calculated = now_datetime

            except Exception as e:
                if self.tracking:
                    self.log_message(f'Error calculating flight path: {str(e)}')
                    print(traceback.format_exc())

        return az, az_rate, el, el_rate, slant_range, range_rate, latitude, longitude, altitude, f1

    def tracking_mode_RA_DEC(self, t):
        '''
        Parameters:
            t (skyfield timescale): Skyfield time

        Returns:
            az (float): Azimuth in degrees 
            el (float): Elevation in degrees
            latitude (float): Subpoint latitude in degrees
            longitude (float): Subpoint longitude in degrees

        NOTE: Currently it the function only works for current time and NOT FOR ANY t.
        This is because it uses self.ra_dec_to_az_el() -> self.get_locale_sidereal_time() -> astropy_time.now()
        If this function should be used for an arbitrary time t, this needs to be adjusted.

        NOTE: IF TRACKING IS TURNED OFF, ALL ERROR MESSAGES ARE GETTING IGNORED! 
        The reason for that is that, if the user has not yet finished typing in all necessary information the program would 
        raise lots of errors. So, the idea is that we just display the data that we can calculate with the information
        that we curretly have. However as soon as tracking is turned on we have to assume that the user has entered all 
        necessary information. No we no longer ignore error messages in order to warn the user, if the given information is
        not valid.
        '''
        try:
            if self.ra_input.text() == '':
                ra_hours = 0
            else:
                ra_hours = float(self.ra_input.text())
            
            if self.dec_input.text() == '':
                dec_degrees = 0
            else:
                dec_degrees = float(self.dec_input.text())
            
            az, el = self.ra_dec_to_az_el(ra_hours, dec_degrees)
        except Exception as e:
            if self.tracking:
                self.log_message(f'Error: {e}')
                print(traceback.format_exc())
            return
        
        earth = 399  # NAIF code for the Earth center of mass
        satellite = position_of_radec(ra_hours, dec_degrees, t=t, center=earth)
        subpoint = wgs84.subpoint(satellite)

        latitude = subpoint.latitude.degrees
        longitude = subpoint.longitude.degrees

        # flight path -----------------------------------------------------------------------------
        self.flight_path = None

        return az, el, latitude, longitude

    def tracking_mode_TLE_OMM(self, t):
        '''
        Parameters:
            t (skyfield timescale): Skyfield time

        Returns:
            az (float): Azimuth in degrees 
            az_rate (float): Azimuth rate in degrees per second
            el (float): Elevation in degrees
            el_rate (float): Elevation rate in degrees per second
            slant_range (float): Distance from antenna to satellite in km
            range_rate (float): Range rate in km/s
            latitude (float): Subpoint latitude in degrees
            longitude (float): Subpoint longitude in degrees
            altitude (float): Altitude of satellite above the ground in km
            f1 (float): Doppler shifted frequency in MHz            

        NOTE: IF TRACKING IS TURNED OFF, (ALMOST) ALL ERROR MESSAGES ARE GETTING IGNORED! 
        The reason for that is that, if the user has not yet finished typing in all necessary information the program would 
        raise lots of errors. So, the idea is that we just display the data that we can calculate with the information
        that we curretly have. However as soon as tracking is turned on we have to assume that the user has entered all 
        necessary information. No we no longer ignore error messages in order to warn the user, if the given information is
        not valid.
        '''
        file_path = self.gp_file_input.text()
        file_ending = file_path.split('.')[-1]

        satellite_name = self.gp_file_satellite_name.text().upper()
        satellite_intl_id = self.gp_file_intl_id.text()
        if self.gp_file_norad_id.text() == '':
            satellite_norad_id = -1
        else:
            satellite_norad_id = int(self.gp_file_norad_id.text())

        satellite = None

        if os.path.isfile(file_path) and (satellite_name != '' or satellite_intl_id != '' or satellite_norad_id != -1):
            if file_ending == 'csv': # OMM ------------------------------------------------

                # find satellite in data
                if satellite_name != '':
                    row = self.omm_df[self.omm_df['OBJECT_NAME'] == satellite_name]
                    if row.empty and self.tracking:
                        self.log_message(f'Could not find {satellite_name} in file {file_path}')

                elif satellite_intl_id != '':
                    row = self.omm_df[self.omm_df['OBJECT_ID'] == satellite_intl_id]
                    if row.empty and self.tracking:
                        self.log_message(f'Could not find {satellite_intl_id} in file {file_path}')
                
                elif satellite_norad_id != -1:
                    row = self.omm_df[self.omm_df['NORAD_CAT_ID'] == satellite_norad_id]
                    if row.empty and self.tracking:
                        self.log_message(f'Could not find {satellite_norad_id} in file {file_path}')

                if not row.empty: # create EarthSatellite
                    fields = row.to_dict(orient='records')[0]
                    satellite = EarthSatellite.from_omm(self.skyfield_ts, fields)

            elif file_ending == 'tle': # TLE ----------------------------------------------

                if satellite_name != '':
                    try:                
                        satellite = self.tle_by_name[satellite_name]
                    except:
                        if self.tracking:
                            self.log_message(f'Could not find {satellite_name} in file {file_path}')
                
                elif satellite_intl_id != '':
                    # adjust to TLE norm for international designator
                    if '-' in satellite_intl_id:
                        satellite_intl_id = satellite_intl_id[2:].replace('-', '')

                    try:                
                        satellite = self.tle_by_intl[satellite_intl_id]
                    except:
                        if self.tracking:
                            self.log_message(f'Could not find {satellite_intl_id} in file {file_path}')
                
                elif satellite_norad_id != -1:
                    try:                
                        satellite = self.tle_by_norad[satellite_norad_id]
                    except:
                        if self.tracking:
                            self.log_message(f'Could not find {satellite_norad_id} in file {file_path}')
                            
            else:
                self.log_message('Invalide file')

            if satellite is not None:
                self.gp_file_satellite = satellite

                # relative position vector
                relative_pos = satellite - self.skyfield_antenna_pos 
                
                # relative position object
                topocentric = relative_pos.at(t)
                satellite = satellite.at(t)

                el, az, slant_range, el_rate, az_rate, range_rate = topocentric.frame_latlon_and_rates(self.skyfield_antenna_pos)

                subpoint = wgs84.subpoint_of(satellite)
                altitude = wgs84.height_of(satellite)

                # units ---------------------------------------------------------------------------
                az = az.degrees
                el = el.degrees
                slant_range = slant_range.km

                az_rate = az_rate.degrees.per_second
                el_rate = el_rate.degrees.per_second
                range_rate = range_rate.km_per_s
                
                latitude = subpoint.latitude.degrees
                longitude = subpoint.longitude.degrees
                altitude = altitude.km

                # doppler shift -------------------------------------------------------------------
                # get frequency from UI
                if self.doppler_initial_freq.text() == '':
                    f0 = 0 
                else:
                    f0 = float(self.doppler_initial_freq.text())
                
                try:
                    f1 = self.doppler_shift(f0, range_rate)
                except Exception as e:
                    if self.tracking:
                        self.log_message(f'Error calculating doppler shift: {str(e)}')
                        print(traceback.format_exc())
        
                # flight path -----------------------------------------------------------------------------
                now_datetime = self.skyfield_time_to_datetime(t)
                if self.last_time_flight_path_got_calculated is not None:
                    delta_t_min = (now_datetime - self.last_time_flight_path_got_calculated).total_seconds() // 60
                else:
                    delta_t_min = self.min_before_recalculate_flight_path
                    self.last_time_flight_path_got_calculated = now_datetime

                if delta_t_min >= self.min_before_recalculate_flight_path:
                    try:
                        flight_path = np.zeros((self.flight_path_steps,2))
                        for i in range(self.flight_path_steps):
                            t = self.datetime_to_skyfield_time(now_datetime + timedelta(minutes=i))
                            satellite = self.gp_file_satellite.at(t)
                            subpoint = wgs84.subpoint_of(satellite)
                            flight_path[i][0] = subpoint.latitude.degrees
                            flight_path[i][1] = subpoint.longitude.degrees
                        
                        self.flight_path = flight_path
                        self.last_time_flight_path_got_calculated = now_datetime

                    except Exception as e:
                        if self.tracking:
                            self.log_message(f'Error calculating flight path: {str(e)}')
                            print(traceback.format_exc())

                return az, az_rate, el, el_rate, slant_range, range_rate, latitude, longitude, altitude, f1

    def tracking_mode_SPICE(self, t):
        '''
        Parameters:
            t (skyfield timescale): Skyfield time

        Returns:
            az (float): Azimuth in degrees 
            az_rate (float): Azimuth rate in degrees per second
            el (float): Elevation in degrees
            el_rate (float): Elevation rate in degrees per second
            slant_range (float): Distance from antenna to satellite in km
            range_rate (float): Range rate in km/s
            latitude (float): Subpoint latitude in degrees
            longitude (float): Subpoint longitude in degrees
            altitude (float): Altitude of satellite above the ground in km
            f1 (float): Doppler shifted frequency in MHz            

        NOTE: IF TRACKING IS TURNED OFF, ALL ERROR MESSAGES ARE GETTING IGNORED! 
        The reason for that is that, if the user has not yet finished typing in all necessary information the program would 
        raise lots of errors. So, the idea is that we just display the data that we can calculate with the information
        that we curretly have. However as soon as tracking is turned on we have to assume that the user has entered all 
        necessary information. No we no longer ignore error messages in order to warn the user, if the given information is
        not valid.
        '''
        if not self.spice_kernels_loaded:
            return
        
        datetime_t = self.skyfield_time_to_datetime(t)
        et = spiceypy.datetime2et(datetime_t)
        satellite_name = self.spice_name.text()
        lat = np.radians(self.antenna_latitude)
        lon = np.radians(self.antenna_longitude)
        alt = self.antenna_altitude/1000

        try:
            # convert lat, lon to xyz -------------------------------------------------------------
            obspos = spiceypy.georec(lon, lat, alt, 6378.1366, 1.0/298.25642)

            # Get the xyz coordinates of the spacecraft relative to observer in MYTOPO frame (Correct for one-way light time and stellar aberration)
            state, _ = spiceypy.spkcpo(satellite_name, et, 'MYTOPO', 'OBSERVER', 'LT+S', obspos, 'EARTH', 'ITRF93')
            
            # Range rate --------------------------------------------------------------------------
            position = np.array(state[:3])  # X, Y, Z
            velocity = np.array(state[3:])  # Vx, Vy, Vz

            # Compute unit line-of-sight vector
            los_unit = position / np.linalg.norm(position)

            # Compute range rate (scalar projection of velocity onto line of sight)
            range_rate = np.dot(velocity, los_unit) # km/s

            # Range, Azimuth, Elevation -----------------------------------------------------------
            slant_range, az, el = spiceypy.recazl(position, azccw=False, elplsz=True)
            az = np.degrees(az)
            el = np.degrees(el)

            # Azimuth and Elevation rates ---------------------------------------------------------
            
            delta_t = 1 # s
            et_future = spiceypy.datetime2et(datetime_t + timedelta(seconds=delta_t))

            # Get state at future time
            state_future, _ = spiceypy.spkcpo(satellite_name, et_future, 'MYTOPO', 'OBSERVER', 'LT+S', obspos, 'EARTH', 'ITRF93')
            position_future = np.array(state_future[:3])

            # Compute future az, el
            _, az_future, el_future = spiceypy.recazl(position_future, azccw=False, elplsz=True)
            az_future = np.degrees(az_future)
            el_future = np.degrees(el_future)

            # Compute rates
            az_rate = (az_future - az) / delta_t  # deg/s
            el_rate = (el_future - el) / delta_t  # deg/s

            # Subpoint, Altitude  -----------------------------------------------------------------
            rot_matrix = spiceypy.pxform('MYTOPO', 'ITRF93', et)

            # Transform the satellite position from MYTOPO to ITRF93
            sat_pos_itrf = rot_matrix @ position

            # Now we use the satellite position in ITRF93 to compute the subpoint
            # Convert rectangular coordinates to geodetic (latitude, longitude, altitude)
            longitude, latitude, altitude = spiceypy.recgeo(sat_pos_itrf, 6378.1366, 1.0/298.25642)
            latitude = np.degrees(latitude)
            longitude = np.degrees(longitude)

        except Exception as e:
            if self.tracking:
                self.log_message(f'Error calculating position with spiceypy: {e}')
                print(traceback.format_exc())
        
        # Doppler Shift ---------------------------------------------------------------
        # get frequency from UI
        if self.doppler_initial_freq.text() == '':
            f0 = 0 
        else:
            f0 = float(self.doppler_initial_freq.text())
        f1 = self.doppler_shift(f0, range_rate)

        # flight path -----------------------------------------------------------------------------
        now_datetime = datetime_t
        if self.last_time_flight_path_got_calculated is not None:
            delta_t_min = (now_datetime - self.last_time_flight_path_got_calculated).total_seconds() // 60
        else:
            delta_t_min = self.min_before_recalculate_flight_path
            self.last_time_flight_path_got_calculated = now_datetime

        if delta_t_min >= self.min_before_recalculate_flight_path:
            try:
                flight_path = np.zeros((self.flight_path_steps,2))
                for i in range(self.flight_path_steps):
                    et = spiceypy.datetime2et(now_datetime + timedelta(minutes=i))
                    state, _ = spiceypy.spkcpo(satellite_name, et, 'MYTOPO', 'OBSERVER', 'LT+S', obspos, 'EARTH', 'ITRF93')
                    position = np.array(state[:3])  # X, Y, Z
                    rot_matrix = spiceypy.pxform('MYTOPO', 'ITRF93', et)
                    sat_pos_itrf = rot_matrix @ position
                    long, lat, altitude = spiceypy.recgeo(sat_pos_itrf, 6378.1366, 1.0/298.25642)

                    flight_path[i][0] = np.degrees(lat)
                    flight_path[i][1] = np.degrees(long)
                
                self.flight_path = flight_path
                self.last_time_flight_path_got_calculated = now_datetime

            except Exception as e:
                if self.tracking:
                    self.log_message(f'Error calculating flight path: {str(e)}')
                    print(traceback.format_exc())

        return az, az_rate, el, el_rate, slant_range, range_rate, latitude, longitude, altitude, f1

    def tracking_mode_AZ_EL(self):
        '''            
        Returns:
            az (float): Azimuth in degrees 
            el (float): Elevation in degrees
        '''
        
        if self.az_input.text() == '':
            az = 0
        else:
            az = float(self.az_input.text())

        if self.el_input.text() == '':
            el = 0
        else:
            el = float(self.el_input.text())
    
        if az < 0 or 360 < az:
            self.log_message('Azimuth need to be between 0° and 360°')
            return
        if el < 0 or 90 < el:
            self.log_message('Elevation need to be between 0° and 90°')
            return
        
        # flight path -----------------------------------------------------------------------------
        self.flight_path = None

        return az, el

    # connected functions -------------------------------------------------------------------------
    def on_tracking_mode_changed(self, index):
        '''
        Parameters:
            index (int): index of satellite in satellite list
        '''
        if self.tracking:
            self.toggle_tracking(False)
            self.log_message('Tracking stopped because tracking methode was changed')
        self.tracking_mode_stack.setCurrentIndex(index)
        self.doppler_initial_freq.setText('0.0')
        self.last_time_flight_path_got_calculated = None

    def on_tracking_mode_list_dropdown_changed(self):
        self.last_time_flight_path_got_calculated = None

    def toggle_tracking(self, checked):
        '''
        Parameters:
            checked (bool): True -> turn tracking on, False -> turn tracking off
        '''

        self.tracking = checked
        if checked:
            self.tracking_btn.setText('Stop Tracking')

            # ensures that the button is checked if the function was not called by the button
            self.tracking_btn.setChecked(True)
        else:
            self.tracking_btn.setText('Start Tracking')

            if self.socket is not None: # send stop command to motors
                self.talk_to_motor_controller('stop')

            # ensures that the button is not checked if the function was not called by the button
            self.tracking_btn.setChecked(False)

    def find_passes(self, return_data=False):
        '''
        Finds the times when the satellite is rising over / setting under the horizon.

        Parameters:
            return_data (bool): if True it will not print to console and return the data. 

        Returns:
            passes (list): (N,2) dim list with AOS datetime and LOS datetime
        '''
        start_time = self.start_time_input.dateTime().toPython()
        start_time = start_time.replace(tzinfo=timezone.utc)
        end_time = self.end_time_input.dateTime().toPython()
        end_time = end_time.replace(tzinfo=timezone.utc)
        if self.min_elevation_input.text() == '':
            min_elevation = 0
        else:
            min_elevation = int(self.min_elevation_input.text().split('°')[0])

        if start_time >= end_time:
            self.log_message('End time must be after start time.')
            return
        
        # convert to UTC if needed
        if self.local_time_radio_button.isChecked():
            start_time = self.local_time_to_UTC(start_time)
            end_time = self.local_time_to_UTC(end_time)

        # find passes -----------------------------------------------------------------------------
        tracking_mode = self.tracking_mode_combo.currentIndex()
        
        if tracking_mode == 0: # List
            current_satellite = self.satellite_list[self.tracking_mode_list_dropdown.currentIndex()]
            self.log_message('Calculating passes...')

            if 'Horizons' in current_satellite['catalogs']: # Horizons
                current_time = start_time
                time_step = timedelta(minutes=10)  # Start with 10-minute steps
                min_time_step = timedelta(minutes=1)  # Minimum step size for precise detection
                max_time_step = timedelta(minutes=10)  # Maximum step size for precise detection
                passes = []
                new_pass = []

                last_el = None

                while current_time <= end_time:
                    '''
                    NOTE: idea for simpler rewrite
                    - go in 10 min steps
                    - if corssing crossing threshold:
                        - go back by 10 min
                        - go in 1 min steps
                        - if corssing crossing threshold:
                            - if last_el < el:
                                - AOS
                            - else:
                                - LOS
                            - go in 10 min steps                 
                    '''
                    _, topocentric = self.calculate_satellite_and_topocentric(current_satellite, self.datetime_to_skyfield_time(current_time))
                    el, _, _, _, _, _ = topocentric.frame_latlon_and_rates(self.skyfield_antenna_pos)
                    el = el.degrees
                                        
                    # If this is the first point
                    if last_el is None:
                        if el >= min_elevation:
                            new_pass.append(current_time)

                        last_el = el
                        current_time += time_step
                        continue
                    
                    # Check if we're crossing the elevation threshold
                    crossing_threshold = (last_el < min_elevation and min_elevation <= el) or (last_el > min_elevation and min_elevation >= el)
                    
                    if crossing_threshold and time_step > min_time_step: # we went too far
                        # Back up and use a smaller step to find the crossing more precisely
                        current_time -= time_step
                        time_step = min_time_step
                    else:
                        # Process the current point
                        if last_el < min_elevation and el >= min_elevation:  # AOS (Acquisition of Signal)
                            new_pass.append(current_time)
                        elif last_el >= min_elevation and el < min_elevation:  # LOS (Loss of Signal)
                            if len(new_pass) == 1:
                                new_pass.append(current_time)
                                passes.append(new_pass)
                                new_pass = []
                            
                        # Use larger steps when we're far from the threshold
                        delta_to_target = abs(el - min_elevation) # deg

                        if delta_to_target > 5:
                            time_step = max_time_step
                        elif delta_to_target > 1:
                            time_step_in_min = time_step.total_seconds()//60
                            rate = abs(el - last_el)/time_step_in_min # deg/min

                            time_step_temp = delta_to_target / rate # min
                            time_step_temp = int(time_step_temp) - 1 # min
                            time_step_temp = max(min_time_step.total_seconds()//60, time_step_temp)
                            time_step_temp = min(max_time_step.total_seconds()//60, time_step_temp)
                            time_step = timedelta(minutes=time_step_temp)
                        else:
                            time_step = min_time_step
                        
                        last_el = el
                        current_time += time_step
                        
                # Handle the case where we end in the middle of a pass
                if len(new_pass) == 1:
                    new_pass.append(current_time)
                    passes.append(new_pass)
            
            else: # CelesTrak
                # convert to Skyfield time
                start_time = self.datetime_to_skyfield_time(start_time)
                end_time = self.datetime_to_skyfield_time(end_time)

                satellite = current_satellite['EarthSatellite']
                times, events = satellite.find_events(self.skyfield_antenna_pos, start_time, end_time, altitude_degrees=min_elevation)

                passes = []
                new_pass = []
                for t, event in zip(times, events):
                    if event == 0: # satellite rises over horizon
                        new_pass.append(t)
                    elif event == 2 and len(new_pass) == 1: # satellite sets under horizon
                        new_pass.append(t)
                        passes.append(new_pass)
                        new_pass = []

                for i in range(len(passes)):
                    passes[i][0] = self.skyfield_time_to_datetime(passes[i][0])
                    passes[i][1] = self.skyfield_time_to_datetime(passes[i][1])

                # convert back to data time
                start_time = self.skyfield_time_to_datetime(start_time)
                end_time = self.skyfield_time_to_datetime(end_time)
        else:
            self.log_message("The 'Find Passes' feature only works for the tracking mode 'List'.")
            return

        # convert back to Local Time if needed
        if self.local_time_radio_button.isChecked():
            start_time = self.UTC_to_local_time(start_time)
            end_time = self.UTC_to_local_time(end_time)

            for i in range(len(passes)):
                passes[i][0] = self.UTC_to_local_time(passes[i][0])
                passes[i][1] = self.UTC_to_local_time(passes[i][1])

        # Print results in console ----------------------------------------------------------------
        if passes:
            if return_data:            
                return passes
            else:
                tz = 'UTC'
                if self.local_time_radio_button.isChecked():
                    tz = 'Local Time'

                plural = 'es'
                if len(passes) == 1:
                    plural = ''

                self.log_message(f'Found {len(passes)} pass{plural} for minimum elevation angle of {min_elevation}°')
                
                for i, p in enumerate(passes):
                    self.log_message(f'Pass {i+1} ------------------------')
                    self.log_message(f'AOS: {p[0].strftime('%H:%M %d.%m.%Y')} {tz}')
                    self.log_message(f'LOS: {p[1].strftime('%H:%M %d.%m.%Y')} {tz}')
        else:
            self.log_message('No passes found for the selected time range')

    def browse_gp_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Select TLE/OMM File', '', 'All Files (*)'
        )
        if file_path:
            self.gp_file_input.setText(file_path)

            file_ending = file_path.split('.')[-1]

            if file_ending == 'csv': # OMM --------------------------------------------------------
                try: # read file
                    self.omm_df = pd.read_csv(file_path)
                except Exception as e:
                    self.log_message(f'Error reading data from file {file_path}: {e}')
                    print(traceback.format_exc())

            elif file_ending == 'tle': # TLE ------------------------------------------------------
                with load.open(file_path) as f:
                    satellites = list(parse_tle_file(f, self.skyfield_ts))
                self.tle_by_name = {sat.name: sat for sat in satellites}
                self.tle_by_norad = {sat.model.satnum: sat for sat in satellites}
                self.tle_by_intl = {sat.model.intldesg: sat for sat in satellites}            
            
            else: # Invalide file -----------------------------------------------------------------
                self.log_message('Invalide file')

    def browse_spice_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Select SPICE Meta Kernel', '', 'All Files (*)'
        )
        if file_path:
            self.spice_input.setText(file_path)

            # Load all kernels from meta-kernel
            try:
                spiceypy.furnsh(file_path)
                self.spice_kernels_loaded = True
            except Exception as e:
                self.log_message(f'Could not load SPICE Kernels: {e}')
                print(traceback.format_exc())

    def update_continuously(self):
        '''
        Gets called all 500 ms (self.update_frequency). It calculates the target azimuth and elevation and
        depending on the tracking option also the range, range rate, altitude, subpoint and doppler shift. All data that
        is available gets displayed on the UI, that includes the current position of the motors. If tracking is turned on 
        the motor steering is enabled too.        

        NOTE: IF TRACKING IS TURNED OFF, ALL ERROR MESSAGES ARE GETTING IGNORED! 
        The reason for that is that, if the user has not yet finished typing in all necessary information the program would 
        raise lots of errors. So, the idea is that we just display the data that we can calculate with the information
        that we curretly have. However as soon as tracking is turned on we have to assume that the user has entered all 
        necessary information. No we no longer ignore error messages in order to warn the user, if the given information is
        not valid.
        '''
        self.UTC_text.setDateTime(QDateTime.currentDateTimeUtc())
        
        try:
            # Tracking Options --------------------------------------------------------------------
            # Get the current tracking option index
            tracking_mode = self.tracking_mode_combo.currentIndex()
            t = self.skyfield_ts.now()
            
            # not all methods return all parameters but the variables need to exist
            az = 0
            az_rate = None
            el = 0
            el_rate = None
            slant_range = None
            range_rate = None
            latitude = None 
            longitude = None
            altitude = None
            f1 = 0

            try:
                if tracking_mode == 0:    # List
                    az, az_rate, el, el_rate, slant_range, range_rate, latitude, longitude, altitude, f1 = self.tracking_mode_List(t)

                elif tracking_mode == 1:  # RA/DEC
                    az, el, latitude, longitude = self.tracking_mode_RA_DEC(t)

                elif tracking_mode == 2:  # TLE/OMM File
                    az, az_rate, el, el_rate, slant_range, range_rate, latitude, longitude, altitude, f1 = self.tracking_mode_TLE_OMM(t)

                elif tracking_mode == 3:  # SPICE
                    az, az_rate, el, el_rate, slant_range, range_rate, latitude, longitude, altitude, f1 = self.tracking_mode_SPICE(t)
                        
                elif tracking_mode == 4:  # AZ/EL
                    az, el = self.tracking_mode_AZ_EL()

            except Exception as e:
                if self.tracking:
                    self.log_message(f'Error calculating satellite data: {e}')
                    print(traceback.format_exc())

            # Correction for not ideal Antenna ----------------------------------------------------
            try:
                az, el = self.correction_matrix(az, el, roll=0, pitch=0, yaw=0)
            except Exception as e:
                if self.tracking:
                    self.log_message(f'Error calculating correction matrix: {e}')
                    print(traceback.format_exc())

            # Update data on UI -------------------------------------------------------------------
            # Target Azimuth and Elevation
            self.target_azimuth.setText(f'{az:.1f}°')
            self.target_elevation.setText(f'{el:.1f}°')
            
            # Doppler Shift
            self.doppler_shifted_freq.setText(f'{f1:.6f}')

            # World Map
            try:
                self.update_map(latitude, longitude, altitude)
            except Exception as e:
                self.log_message(f'Error updating Map: {str(e)}')
                print(traceback.format_exc())

            # Altitude
            if altitude is not None:
                self.altitude_text.setText(f'{altitude:.0f} km')
            else:
                self.altitude_text.setText('0 km')

            # Range
            if slant_range is not None:
                self.range_text.setText(f'{slant_range:.0f} km')
            else:
                self.range_text.setText('0 km')

            # Range Rate
            if range_rate is not None:
                self.range_rate_text.setText(f'{range_rate:.3f} km/s')
            else:
                self.range_rate_text.setText('0 km/s')
            # -------------------------------------------------------------------------------------

            # manual offset
            az += self.azimuth_offset.value()
            el += self.elevation_offset.value()

            # start tacking at AOS
            if self.start_tracking_at_AOS_btn.isChecked():
                if not self.tracking and el > 0 and tracking_mode in [0,2,3]:
                    self.toggle_tracking(True)
                    self.start_tracking_at_AOS_btn.setChecked(False)
                    self.log_message('Tracking was started automatically at expected AOS.')

            # stop tracking when satellite is under the horizon
            if self.tracking and el < 0:
                self.toggle_tracking(False)
                self.log_message('Tracking was stopped because the satellite is under the horizon.')

            # Motors ------------------------------------------------------------------------------
            if self.socket is not None:
                # get current position from antenna
                current_az, current_el = self.talk_to_motor_controller('status')

                self.current_azimuth.setText(f'{current_az:.1f}°')
                self.current_elevation.setText(f'{current_el:.1f}°')

                if self.should_update_motors(current_az, current_el, az, el) and self.tracking:
                    # calculate target position based on angular rate
                    now = self.skyfield_time_to_datetime(t)
                    if az_rate is not None and el_rate is not None:
                        if self.last_time_motor_got_updated is not None:
                            delta_t = (now - self.last_time_motor_got_updated).total_seconds()
                            az += az_rate*delta_t
                            el += el_rate*delta_t
                    self.last_time_motor_got_updated = now

                    az = np.clip(az, 0, 360)
                    el = np.clip(el, 0, 90)
                    self.talk_to_motor_controller('set', az, el)
            
        except Exception as e:
            if self.tracking:
                self.log_message(f'Error: {str(e)}')
                print(traceback.format_exc())        

    def UTC_local_time_button_func(self):
        '''
        Changes if the time in the find passes widget gets displayed in local time or UTC
        '''
        if self.utc_radio_button.isChecked():
            self.start_time_input.setTimeZone(QTimeZone(b'UTC'))
            self.end_time_input.setTimeZone(QTimeZone(b'UTC'))
        else:
            # convert str to bytes
            tz = self.local_tz.encode()
            self.start_time_input.setTimeZone(QTimeZone(tz))
            self.end_time_input.setTimeZone(QTimeZone(tz))

        self.start_time_input.setDateTime(QDateTime.currentDateTime())
        self.end_time_input.setDateTime(QDateTime.currentDateTime().addDays(1))

    def add_satellite_to_list(self):
        if self.gp_file_satellite is not None:
            name = self.gp_file_satellite.name
            norad_id = self.gp_file_satellite.model.satnum
            intl_id = self.gp_file_satellite.model.intldesg
            if self.doppler_initial_freq.text() == '':
                f0 = 0
            else:
                f0 = float(self.doppler_initial_freq.text())

            # change format from 25066A to 2025-066A
            oldest_launch_year = 64 # TODO? automate
            if int(intl_id[:2]) < oldest_launch_year:
                intl_id = f'20{intl_id[:2]}-{intl_id[2:]}'
            else:
                intl_id = f'19{intl_id[:2]}-{intl_id[2:]}'

            new_entry = {
                "name": name,
                "catalogs": {
                    "NORAD": norad_id,
                    "Int'l": intl_id
                },
                "frequency": f0
            }

            json_file = os.path.join('Main', 'config', 'satellite_list.json')

            # Load existing data or start with an empty list
            try:
                with open(json_file, 'r') as file:
                    data = json.load(file)
            except (FileNotFoundError, json.JSONDecodeError):
                data = []

            # Check if the satellie is already in json
            if not any(entry.get('name') == new_entry['name'] for entry in data):
                data.append(new_entry)
                with open(json_file, 'w') as file:
                    json.dump(data, file, indent=4)
                self.log_message(f'{name} was added to the list.')
                new_entry['EarthSatellite'] = self.gp_file_satellite
                self.satellite_list.append(new_entry)
                self.tracking_mode_list_dropdown.addItems([name])
            else:
                self.log_message(f'{name} is already in the list.')

        else:
            self.log_message('No satellite selected!')

    def visualise_next_pass(self):
        data = self.find_passes(return_data=True)

        start_time = data[0][0]
        end_time = data[0][1]
        delta_t = int((end_time - start_time).total_seconds())

        current_satellite = self.satellite_list[self.tracking_mode_list_dropdown.currentIndex()]
        
        # There is absolutly not need to plot more then 500 points.
        # Therefore we can optimise the calculation by reducing the 
        # amount of calculated points
        step_size = 1 # seconds per step
        while delta_t > 500:
            delta_t = delta_t // 10
            step_size *= 10
        
        data = np.zeros((delta_t,2))
        tracking_mode = self.tracking_mode_combo.currentIndex()
        if tracking_mode == 0: # List
            for i in range(delta_t):
                current_time = start_time + timedelta(seconds=i*step_size)

                t = self.datetime_to_skyfield_time(current_time)
                _, topocentric = self.calculate_satellite_and_topocentric(current_satellite, t)
                el, az, _, _, _, _ = topocentric.frame_latlon_and_rates(self.skyfield_antenna_pos)
                data[i,0] = az.degrees
                data[i,1] = el.degrees

        self.plot_window = NexPassVisualisationWindow()
        self.plot_window.plot(data, start_time, end_time, self.utc_radio_button.isChecked())
        self.plot_window.show()

    # get & load satellite data -------------------------------------------------------------------
    def get_satellite_names_from_file(self):
        config_file_path = os.path.join('Main', 'config', 'satellite_list.json')
        try:
            with open(config_file_path, 'r') as file:
                satellite_data = json.load(file)
                satellite_names = [satellite['name'] for satellite in satellite_data] 
                return satellite_names
        except Exception as e:
            self.log_message(f'Error reading satellite config file: {e}')
            print(traceback.format_exc())
            return []
        
    def get_satellites_from_file(self):
        config_file_path = os.path.join('Main', 'config', 'satellite_list.json')
        try:
            with open(config_file_path, 'r') as file:
                satellite_data = json.load(file)
                return satellite_data
        except Exception as e:
            if hasattr(self, 'console'):
                self.log_message(f'Error reading satellite list file: {e}')
                print(traceback.format_exc())
                return []
            else:
                print(e)
                print('>>>>>> The program could not start because of a problem with the satellite list file <<<<<<')

    def load_satellite_metadata(self):
        config_file_path = os.path.join('Main', 'data', 'metadata.json')
        try:
            with open(config_file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            self.log_message(f'Error reading satellite metadata file: {e}')
            print(traceback.format_exc())
            return {}
        
    def save_satellite_metadata(self):
        config_file_path = os.path.join('Main', 'data', 'metadata.json')
        try:
            with open(config_file_path, 'w') as file:
                json.dump(self.satellite_metadata, file, indent=4)
        except Exception as e:
            self.log_message(f'Error saving satellite metadata file: {e}')
            print(traceback.format_exc())
        
    def load_all_satellite_data(self, celestrak_only=False, ID=None):
        '''
        Loads locally saved data for all satellies that are specified in the satellite list

        Parameters:
            celestrak_only (bool): Flag if we want to only (re)load the CelesTrak data
            ID (int): Id of spacecraft. When we want to only (re)load the Horizon data for a specific spacecraft
        
        '''

        file_path = os.path.join('Main', 'data', 'CelesTrak', 'all_active_satellites.csv')
        self.all_CelesTrak_satellites_df = pd.read_csv(file_path)

        for current_satellite in self.satellite_list:
            # Horizons --------------------------
            if 'Horizons'  in current_satellite['catalogs']:
                if celestrak_only: # after CelesTrak data was updated
                    continue       # there is no need to reload Horizon data

                spacecraft_id = current_satellite['catalogs']['Horizons']
                if ID is not None and ID != spacecraft_id: # after Horizon data was updated
                    continue                               # there is no need to update all Horizon satellites
                
                file_name = f'spacecraft_data_({spacecraft_id}).csv'
                file_path = os.path.join('Main', 'data', 'Horizons', file_name)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    current_satellite['df'] = df
                else:
                    self.log_message(f'File does not exist: {file_path}')
                    try:
                        # download data
                        self.log_message(f'Downloading new data for Spacecraft {spacecraft_id}...')
                        self.query_horizons_api(spacecraft_id)

                        # and try again
                        df = pd.read_csv(file_path)
                        current_satellite['df'] = df
                    except Exception as e:
                        self.log_message(f'Error downloading and reading data: {str(e)}')
                        print(traceback.format_exc())

            # CelesTrak -------------------------       
            elif "Int'l" in current_satellite['catalogs']:
                satelltie_id = current_satellite['catalogs']["Int'l"]
                row = self.all_CelesTrak_satellites_df[self.all_CelesTrak_satellites_df['OBJECT_ID'] == satelltie_id]
                if not row.empty:
                    fields = row.to_dict(orient='records')[0]
                    satellite = EarthSatellite.from_omm(self.skyfield_ts, fields)
                    current_satellite['EarthSatellite'] = satellite
                else:
                    self.log_message(f'Data for {current_satellite} is empty.')
                            
            # if data does not contain Int'l id, try NORAD id
            elif 'NORAD' in current_satellite['catalogs']:
                satelltie_id = current_satellite['catalogs']['NORAD']
                fields = self.all_CelesTrak_satellites_df[self.all_CelesTrak_satellites_df['NORAD_CAT_ID'] == satelltie_id]
                if not fields.empty:
                    satellite = EarthSatellite.from_omm(self.skyfield_ts, fields)
                    current_satellite['EarthSatellite'] = satellite
                else:
                    self.log_message(f'Data for {current_satellite} is empty.')

            else:
                self.log_message(f'Could not find satellite in data {current_satellite}')

    def query_celestrak_api(self):
        file_path = os.path.join('Main', 'data', 'CelesTrak', 'all_active_satellites.csv')
        url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=csv'

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            with open(file_path, 'wb') as file:
                file.write(response.content)
            self.log_message(f'Data downloaded and saved to {file_path}')

        except requests.exceptions.ConnectionError:
            self.log_message('Error downloading data from CelesTrak: No internet connection.')
            print('No internet connection.')
            return False
        
        except Exception as e:
            self.log_message(f'Error downloading data from CelesTrak: {e}')
            print(traceback.format_exc())
            return False
        
        # Update metadata
        now_utc = self.utc_now()
        valid_from = self.utc_now()
        valid_until = self.utc_now() + timedelta(hours=2)
        self.satellite_metadata['CelesTrak']['last download'] = now_utc.isoformat()
        self.satellite_metadata['CelesTrak']['valid from'] = valid_from.isoformat()
        self.satellite_metadata['CelesTrak']['valid until'] = valid_until.isoformat()
        self.save_satellite_metadata()

    def query_horizons_api(self, spacecraft_id):
        '''
        Query the JPL Horizons API for spacecraft state vectors.

        Parameters:
            spacecraft_id (int): id of spacecraft of celestial body in Horizons (JPL) catalog
        '''
        # Base URL for the JPL Horizons System
        url = 'https://ssd.jpl.nasa.gov/api/horizons.api'
        
        start_date = (self.utc_now() - timedelta(days=0.5)).strftime('%Y-%m-%dT%H:%M:%S')
        end_date = (self.utc_now() + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S')
        
        # Define parameters for the query
        params = {
            'format': 'json',               # Return results in JSON format
            'COMMAND': str(spacecraft_id),  # The ID of the spacecraft
            'OBJ_DATA': 'NO',               # Include object data
            'MAKE_EPHEM': 'YES',            # Generate ephemeris
            'EPHEM_TYPE': 'VECTORS',        # Vector table
            'CENTER': '@0',                 # Sun barycenter as origin
            'REF_PLANE': 'FRAME',           # Reference plane
            'REF_SYSTEM': 'ICRF',           # ICRF reference frame
            'VEC_TABLE': '2',               # State vectors
            'VEC_CORR': 'LT+S',             # Vector (light time) correction
            'START_TIME': start_date,       # Start date
            'STOP_TIME': end_date,          # End date
            'STEP_SIZE': '1min',            # Step size
            'CSV_FORMAT': 'YES',            # Data in CSV format
            'VEC_DELTA_T': 'YES',           # Differnece between TDB and UTC
        }
        
        try:
            # Make the request to JPL Horizons API
            response = requests.get(url, params=params)
            
            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                if 'error' in data:
                    print(f'Error query the Horizons API: {data['error']}')
                    return False
                self.save_horizons_results(data, spacecraft_id)
                return True
            else:
                print(f'Error query the Horizons API: API request failed with status code {response.status_code}')
                print(response.text)
                return False
        
        except requests.exceptions.ConnectionError:
            self.log_message('Could not connect to Horizons API: No internet connection.')
            print('No internet connection.')
            return False

        except Exception as e:
            self.log_message(f'Could not get data from Horizons API: {str(e)}')
            print(traceback.format_exc())
            return False

    def process_horizons_output(self, data):
        '''
        Extract CSV from Horizons (JPL) output and add UTC column.

        Parameters:
            data (dir): JSON with the data from Horizons
        '''
        result_text = data['result']
        csv_start = result_text.find('$$SOE')
        csv_end = result_text.find('$$EOE')
        
        if csv_start == -1 or csv_end == -1:
            self.log_message('CSV section not found in JPL output')
            return
        
        csv_data = result_text[csv_start+5:csv_end].strip()
        lines = csv_data.splitlines()
        
        processed_data = []
        for line in lines:
            colums = line.split(',')
            # clean up data -------------------------
            # convert to numbers
            colums_with_numbers = [0,2,3,4,5,6,7,8]
            for i in colums_with_numbers:
                colums[i] = float(colums[i])
            
            # remove empty element
            # Method 1: Last element is empty because of how it was created
            colums.pop()

            # Method 2: More robust, but more expensive
            # colums = list(filter(lambda a: a != '', colums)) 
            # ---------------------------------------
            jd_tdb = colums[0]
            delta_t = colums[2]
            jd_utc = self.convert_tdb_to_utc(jd_tdb, delta_t)
            dt_utc = datetime(2000, 1, 1, 12) + timedelta(days=jd_utc - 2451545.0)
            colums.append(dt_utc.isoformat())  # Append UTC date
            processed_data.append(colums)
        
        return processed_data

    def save_horizons_results(self, data, spacecraft_id):
        '''
        Save Horizons data to CSV and updates metadata
        
        Parameters:
            data (dir): JSON with the data from Horizons
            spacecraft_id (int): id of spacecraft of celestial body in Horizons (JPL) catalog
        '''

        output_dir = os.path.join('Main', 'data', 'Horizons')

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if 'result' in data:            
            # Extract CSV data
            processed_data = self.process_horizons_output(data)
            
            # Convert to DataFrame and save
            headers = ['JDTDB', 'Calendar Date (TDB)', 'delta-T', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'Calendar Date (UTC)']
            df = pd.DataFrame(processed_data, columns=headers)
            
            csv_path = os.path.join(output_dir, f'spacecraft_data_({spacecraft_id}).csv')
            df.to_csv(csv_path, index=False)
            self.log_message(f'Data saved to {csv_path}')

            # Update metadata
            now_utc = self.utc_now()
            valid_from = self.utc_now() - timedelta(days=1)
            valid_until = self.utc_now() + timedelta(days=1)
            if spacecraft_id not in self.satellite_metadata['Horizons']:
                self.satellite_metadata['Horizons'][f'{spacecraft_id}'] = {
                    'last download' : '',
                    'valid from' : '',
                    'valid until' : ''
                }
            self.satellite_metadata['Horizons'][f'{spacecraft_id}']['last download'] = now_utc.isoformat()
            self.satellite_metadata['Horizons'][f'{spacecraft_id}']['valid from'] = valid_from.isoformat()
            self.satellite_metadata['Horizons'][f'{spacecraft_id}']['valid until'] = valid_until.isoformat()
            self.save_satellite_metadata()

    # helper functions ----------------------------------------------------------------------------
    def motor_controller_connect(self):
        '''
        Establish a persistent connection to the motor controller, if available.
        '''

        host = self.motor_controller_IP
        port = self.motor_controller_port

        self.log_message(f'Trying to connect to {host}:{port}...')
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)  # Set a 5-second timeout for operations
            s.connect((host, port))

            self.socket = s  # Store socket
            self.log_message(f'Successfully connected to {host}:{port}')
        except:
            self.socket = None
            self.log_message(f'Could not connect to {host}:{port}')

    def motor_controller_establish_connection(self):
        '''
        Check connection in a separate thread and establishes a socket it sucessfull.
        Using a sperarte thread means that the main program is not block by the time out.
        '''
        thread = threading.Thread(target=self.motor_controller_connect, daemon=True)
        thread.start()

    def motor_controller_close_connection(self):
        '''
        Close the socket connection properly.
        '''

        if self.socket:
            self.socket.close()
            self.socket = None
            self.log_message('Connection closed.')

    def talk_to_motor_controller(self, command, az=0, el=0):
        '''
        Get the current position from the SPID motor controller using TCP socket.
        
        Parameters:
            comand (str): stop, status or set
            az (float): azimuth in degrees
            el (float): elevation in degrees
            
        Returns:
            azimuth (float): azimuth in degrees (only when command is 'status')
            elevation (float): elevation in degrees (only when command is 'status')

        ---------------------- packet (13 bytes) ----------------------
        Format: [START, H1, H2, H3, H4, PH, V1, V2, V3, V4, PV, K, END]
        
        S     : Start byte. This is always 0x57 ('W')
        H1-H4 : Azimuth as ASCII characters 0-9
        PH    : Azimuth resolution in steps per degree (ignored!)
        V1-V4 : Elevation as ASCII characters 0-9
        PV    : Elevation resolution in steps per degree (ignored!)
        K     : Command (0x0F=stop, 0x1F=status, 0x2F=set)
        END   : End byte. This is always 0x20 (space)
        ---------------------------------------------------------------
        '''
        try:
            if self.socket is not None:
                if command == 'stop':            
                    packet = bytearray(13)
                    packet[0] = 0x57    # Start byte 'W'
                    packet[11] = 0x0F   # K (command) byte
                    packet[12] = 0x20   # End byte (space)
                
                    # Send the packet
                    self.socket.sendall(packet)

                elif command == 'set':
                    packet = self.create_set_position_packet(az, el)
                    
                    # Send the packet
                    self.socket.sendall(packet)
                
                elif command == 'status':
                    packet = bytearray(13)
                    packet[0] = 0x57    # Start byte 'W'
                    packet[11] = 0x1F   # K (command) byte
                    packet[12] = 0x20   # End byte (space)
                
                    # Send the packet
                    self.socket.sendall(packet)

                    # Wait until I get a response or socket times out
                    # Read response (12 bytes for Rot2Prog, 5 bytes for Rot1Prog)
                    response = self.socket.recv(12)

                    # If we got less than 12 bytes, maybe it's a Rot1Prog (5 bytes)
                    if len(response) < 12 and len(response) >= 5:
                        # Rot1Prog response format:
                        # [0x57, H1, H2, H3, 0x20]
                        azimuth = response[1] * 100 + response[2] * 10 + response[3] - 360
                        return azimuth, 0
                        
                    elif len(response) >= 12:
                        # Rot2Prog response format:
                        # [0x57, H1, H2, H3, H4, PH, V1, V2, V3, V4, PV, 0x20]
                        azimuth = response[1] * 100 + response[2] * 10 + response[3] + response[4] / 10 - 360
                        elevation = response[6] * 100 + response[7] * 10 + response[8] + response[9] / 10 - 360
                        return azimuth, elevation
                    
                    else:
                        raise TimeoutError(f'Received only {len(response)} bytes, expected 5 or 12')
                    
                else:
                    print(f'invalide command: {command}')
        except:
            '''
            Attempt to reconnect after disconnection. 
            If the motor controller responds within 5 seconds, 
            a new connection will be established. Otherwise, 
            it will be considered that the controller has completely disconnected 
            and no further communication attempt will be made.
            '''
            self.log_message('Disconnected from motor controller. Trying to reconnect...')
            self.motor_controller_close_connection()
            self.motor_controller_establish_connection()
            return 0, 0

    def create_set_position_packet(self, azimuth, elevation, az_resolution=10, el_resolution=10):
        '''
        Create a command packet to set the position for a Rot2Prog controller.
        
        Parameters:
            azimuth (float): Azimuth in degrees (0-360)
            elevation (float): Elevation in degrees
            az_resolution (int): Azimuth resolution in steps per degree
            el_resolution (int): Elevation resolution in steps per degree
            
        Returns:
            bytearray: 13-byte command packet
        '''
        # Create command packet (13 bytes)
        packet = bytearray(13)
        
        # Start byte
        packet[0] = 0x57  # 'W'
        
        # Azimuth encoding
        az_steps = int((360 + azimuth) * az_resolution)
        az_str = f'{az_steps:04d}' # formats the steps count as a 4-digit string with leading zeros

        # encode each digit as its ASCII value
        packet[1] = ord(az_str[0])  # Thousands H1
        packet[2] = ord(az_str[1])  # Hundreds  H2
        packet[3] = ord(az_str[2])  # Tens      H3
        packet[4] = ord(az_str[3])  # Ones      H4
        
        # Azimuth resolution
        packet[5] = az_resolution
        
        # Elevation encoding
        el_steps = int((360 + elevation) * el_resolution)
        el_str = f'{el_steps:04d}'
        packet[6] = ord(el_str[0])  # Thousands V1
        packet[7] = ord(el_str[1])  # Hundreds  V2
        packet[8] = ord(el_str[2])  # Tens      V3
        packet[9] = ord(el_str[3])  # Ones      V4
        
        # Elevation resolution
        packet[10] = el_resolution
        
        # Command byte (0x2F for SET)
        packet[11] = 0x2F
        
        # End byte
        packet[12] = 0x20  # space
        
        return packet

    def update_map(self, latitude, longitude, altitude):
        '''
        Parameters:
            latitude (float): Latitude in degrees
            longitude (float): Longitude in degrees
            altitude (float): Altitude in km
        '''
        self.map_ax.clear() # Clear the previous plot but keep the axis (image)
        
        # if there is not satellie to plot, the flight plan must be old
        if latitude is None or longitude is None or altitude is None:
            self.flight_path = None 
        
        img_extent = (-180, 180, -90, 90)
        self.map_ax.imshow(self.earth_img, origin='upper', extent=img_extent, transform=self.map_projection)
        
        # plot antenna 
        self.map_ax.plot(
            self.antenna_longitude, 
            self.antenna_latitude, 
            'o', 
            color='cyan', 
            markersize=2, 
            transform=self.map_projection
        )

        # plot flight path
        if self.flight_path is not None:

            # Split flight path in multiple paths when we go off the map
            diffs = np.abs(np.diff(self.flight_path[:, 1]))
            jump_indices = np.where(diffs > 180)[0]
            
            # Split the array at these indices
            flight_paths = []
            start_idx = 0
            
            for idx in jump_indices:
                flight_paths.append(self.flight_path[start_idx:idx+1])
                start_idx = idx + 1
            
            # Add the last segment
            if start_idx < len(self.flight_path):
                flight_paths.append(self.flight_path[start_idx:])

            for path in flight_paths:
                self.map_ax.plot(
                    path[:,1],
                    path[:,0],
                    transform=self.map_projection,
                    color= 'orange'
                )
                
        # plot satellite 
        if latitude is not None and longitude is not None:
            self.map_ax.plot(
                longitude, 
                latitude, 
                'o', 
                color='red', 
                markersize=2, 
                transform=self.map_projection
            )

        # plot circle
        if altitude is not None:
            earth_radius = 6378 # km

            theta = np.arccos(earth_radius/(altitude + earth_radius))
            radius = earth_radius * theta

            self.plot_geodesic_circle(self.map_ax, longitude, latitude, radius, color='red', linewidth=1)

        # Grid
        gl = self.map_ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
                
        # Remove margins to maximize map area
        self.map_figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)  # minimize margins
    
        self.map_figure.tight_layout()
        self.map_canvas.draw()

    def plot_geodesic_circle(self, ax, lon, lat, radius_km, **kwargs):
        '''
        Parameters:
            ax (axis): Axis of figure
            lon (float): longnitude of circle center
            lat (float): latitude of circle center
            radius_km (float): radius of circle
            **kwargs (any): arguments regarding the style like color, linewidth, etc...
        '''
        # Create a geodesic circle
        geod = geodesic.Geodesic()
        circle_points = geod.circle(lon, lat, radius_km * 1000, n_samples=100)
            
        # Extract coordinates
        circle_lons = circle_points[:, 0]
        circle_lats = circle_points[:, 1]
        
        # Plot the circle
        ax.plot(circle_lons, circle_lats, transform=ccrs.Geodetic(), **kwargs)
        
        # close the circle
        start_point = circle_points[0]
        end_point = circle_points[-1]

        # Debugging -----------------------------
        # ax.scatter(start_point[0], start_point[1], color='gray')
        # ax.scatter(end_point[0], end_point[1], color='black')
        # ---------------------------------------

        # if it goes over the 180° meridian plot in 2 parts
        if abs(start_point[0] - end_point[0]) > 180:
            lons = [start_point[0], 180]
            lats = [start_point[1], start_point[1]]
            ax.plot(lons, lats, transform=ccrs.Geodetic(), **kwargs)

            lons = [-180, end_point[0]]
            lats = [end_point[1], end_point[1]]
            ax.plot(lons, lats, transform=ccrs.Geodetic(), **kwargs)
        else:
            lons = [start_point[0], end_point[0]]
            lats = [start_point[1], end_point[1]]
            ax.plot(lons, lats, transform=ccrs.Geodetic(), **kwargs)
    
    
        # Create a polygon for filling
        fill_color = 'red'
        fill_alpha = 0.1

        # Handle the case where the circle crosses the dateline
        if abs(start_point[0] - end_point[0]) > 180:
            try:
                # Split the circle into two parts at the dateline
                split_idx = np.where(np.abs(np.diff(circle_lons)) > 180)[0][0]
                
                # First part (before the dateline)
                poly1_lons = circle_lons[:split_idx+1]
                poly1_lats = circle_lats[:split_idx+1]
                
                # Second part (after the dateline)
                poly2_lons = circle_lons[split_idx+1:]
                poly2_lats = circle_lats[split_idx+1:]
                
                # Create and add the first polygon
                poly1_xy = np.column_stack([poly1_lons, poly1_lats])
                poly1 = Polygon(poly1_xy, closed=True, facecolor=fill_color, alpha=fill_alpha, 
                            transform=ccrs.Geodetic(), **{k:v for k,v in kwargs.items() if k not in ['color', 'linestyle', 'linewidth']})
                ax.add_patch(poly1)
                
                # Create and add the second polygon
                poly2_xy = np.column_stack([poly2_lons, poly2_lats])
                poly2 = Polygon(poly2_xy, closed=True, facecolor=fill_color, alpha=fill_alpha,
                            transform=ccrs.Geodetic(), **{k:v for k,v in kwargs.items() if k not in ['color', 'linestyle', 'linewidth']})
                ax.add_patch(poly2)
            except:
                '''
                There is edge case when the satellite is close to a pole. 
                
                    split_idx = np.where(np.abs(np.diff(circle_lons)) > 180)[0][0]
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
                IndexError: index 0 is out of bounds for axis 0 with size 0
                
                It only concerns the drawing of the polygon. 
                There is not visible mistake or artefact on the map, therefore we can catch the error here and ignore it.
                '''
                pass  
        else:
            # Create a simple polygon if the circle doesn't cross the dateline
            poly_xy = np.column_stack([circle_lons, circle_lats])
            poly = Polygon(poly_xy, closed=True, facecolor=fill_color, alpha=fill_alpha,
                        transform=ccrs.Geodetic(), **{k:v for k,v in kwargs.items() if k not in ['color', 'linestyle', 'linewidth']})
            ax.add_patch(poly)
    
    def log_message(self, message):
        '''
        Parameters
            message (string): message
        '''
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.console.append(f'[{timestamp}] {message}')

        # ensure that message gets shown in console NOW
        QApplication.processEvents()

    def keyPressEvent(self, event):
        '''
        Parameters:
            event (PySide6.QtGui.QKeyEvent): event
        '''
        
        # Handle arrow key presses for azimuth and elevation offset
        if event.key() == Qt.Key_Left or event.key() == Qt.Key_A:
            current = self.azimuth_offset.value()
            self.azimuth_offset.setValue(current - 0.1)
            event.accept()  # Mark event as handled
        elif event.key() == Qt.Key_Right or event.key() == Qt.Key_D:
            current = self.azimuth_offset.value()
            self.azimuth_offset.setValue(current + 0.1)
            event.accept()  # Mark event as handled
        elif event.key() == Qt.Key_Up or event.key() == Qt.Key_W:
            current = self.elevation_offset.value()
            self.elevation_offset.setValue(current + 0.1)
            event.accept()  # Mark event as handled
        elif event.key() == Qt.Key_Down or event.key() == Qt.Key_S:
            current = self.elevation_offset.value()
            self.elevation_offset.setValue(current - 0.1)
            event.accept()  # Mark event as handled
        elif event.key() == Qt.Key_Space:
            self.toggle_tracking(not self.tracking)
            event.accept()  # Mark event as handled
        elif event.key() == Qt.Key_Escape:
            self.clearFocus() # rest focuse because being focused on input field can break hotkeys
            self.setFocus()
            event.accept()  # Mark event as handled
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        '''
        Parameters:
            event (PySide6.QtGui.QKeyEvent): event
        '''

        self.motor_controller_close_connection()
                
        print('Satellite Tracker was closed')
        event.accept()  # Ensures the window closes properly

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SatelliteTrackerApp()
    window.show()

    sys.exit(app.exec())
