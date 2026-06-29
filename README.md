# Satellite Tracker

## OLD VERSION!!! USE VERISON 3 (coming soon)

The Department of Physics of the University of Zurich is presently installing a new radio telescope. This antenna should serve educational purposes, giving students the ability to gain first-hand experience in data collection and analysing real astronomical signals. It is meant to be used by high school students and university students and will be able to observe satellites, both in orbit around Earth, as well as spacecraft in deep space and celestial targets like pulsars.

Therefore, an easy-to-use control software is needed that can track targets and orient the antenna automatically with the push of a button.

This thesis aims to develop and test a custom control software for this new radio telescope, which can track deep space and celestial targets as well as Earth satellites and communicate directly with the motor controller of the radio telescope.

[Read the full thesis (PDF)](./Bachelor_Thesis.pdf)

## Dependencies

This project requires the following Python packages:

- numpy
- pandas
- matplotlib
- cartopy
- spiceypy
- requests
- python-dotenv
- astropy
- skyfield
- PySide6

You can install them using pip:

Math and data packages
```bash
pip install numpy pandas matplotlib requests python-dotenv
```

Astronomy packages
```bash
pip install astropy skyfield spiceypy
```

UI and mapping packages
```bash
pip install PySide6 cartopy
```

# Patch Notes
## Version 3
Coming soon.

## Version 1.2
By default the `Start Tracking at AOS` button will be unchecked as soon as the tracking starts. This can be turned off by setting `AUTO_UNCHECK_START_TRACKING_AT_AOS_BTN` in the `config.json` to `False`.
NOTE: Stoping the tracking maually will also uncheck the `Start Tracking at AOS` button, in order to prevent immediate restart of tracking.

## Version 1.1
During the Artemis II mission, a mismatch was identified between the program's AZ/EL calculations based on Horizons data and the AZ/EL data provided by Horizons itself. An option to use the Horizons AZ/EL data directly has been added. Specifically, the following values are overwritten when using the **Horizons Directly** option:

- Azimuth
- Elevation
- Range
- Range Rate

This ensures that the most important features (antenna pointing and Doppler shift calculations) align with the data calculated by Horizons. But you will still have slightly incorrect visualizations, such as the ground track and footprint, since they don't use the direct data.

**Note:** There is a small observed difference between the values on the Horizons website and those retrieved via the Horizons API. For Azimuth, this is <0.1°, and for Elevation, it is <0.5°.
