import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
from ag_models_wrappers.process_ibm_risk import get_weather, get_ibm_weather


class TestGetWeather(unittest.TestCase):

    @patch('ag_models_wrappers.process_ibm_risk.get_ibm_weather')
    @patch('requests.get')
    def test_get_weather_success(self, mock_requests_get, mock_get_ibm_weather):
        # Mock response data for the IBM API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "validTimeUtc": ["2024-12-01T00:00:00Z", "2024-12-01T01:00:00Z"],
            "temperature": [22.5, 23.0],
            "temperatureDewPoint": [10.0, 9.5],
            "relativeHumidity": [80, 82],
            "precip1Hour": [0.1, 0.0],
            "windSpeed": [3.0, 3.5],
        }

        mock_requests_get.return_value = mock_response
        mock_get_ibm_weather.return_value = pd.DataFrame({
            'validTimeUtc': ["2024-12-01T00:00:00Z", "2024-12-01T01:00:00Z"],
            'temperature': [22.5, 23.0],
            'temperatureDewPoint': [10.0, 9.5],
            'relativeHumidity': [80, 82],
            'precip1Hour': [0.1, 0.0],
            'windSpeed': [3.0, 3.5],
        })

        # Call the function
        lat, lng, end_date = 40.0, -75.0, '2024-12-02'
        data = get_weather(lat, lng, end_date)

        self.assertIsNotNone(data)
        self.assertIn('hourly', data)
        self.assertIn('daily', data)
        self.assertEqual(len(data['hourly']), 2)
        self.assertEqual(data['hourly']['temperature'][0], 22.5)

    @patch('ag_models_wrappers.process_ibm_risk.get_ibm_weather')
    @patch('requests.get')
    def test_get_weather_empty_response(self, mock_requests_get, mock_get_ibm_weather):
        # Mock response data for an empty API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}

        mock_requests_get.return_value = mock_response
        mock_get_ibm_weather.return_value = pd.DataFrame()

        # Call the function
        lat, lng, end_date = 40.0, -75.0, '2024-12-02'
        data = get_weather(lat, lng, end_date)

        self.assertIsNone(data['hourly'])
        self.assertIsNone(data['daily'])

    @patch('ag_models_wrappers.process_ibm_risk.get_ibm_weather')
    @patch('requests.get')
    def test_get_weather_api_error(self, mock_requests_get, mock_get_ibm_weather):
        mock_response = MagicMock()
        mock_response.status_code = 500  # Internal server error
        mock_requests_get.return_value = mock_response

        mock_get_ibm_weather.return_value = pd.DataFrame()  # No data case

        lat, lng, end_date = 40.0, -75.0, '2024-12-02'
        data = get_weather(lat, lng, end_date)

        # Assertions
        self.assertIsNone(data['hourly'])
        self.assertIsNone(data['daily'])


if __name__ == '__main__':
    unittest.main()