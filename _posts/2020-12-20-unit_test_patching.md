# Unit Testing: Patching

In this post we'll go over how to use patching (aka mocking) to unit test functions which connect to external data sources. It assumes prior basic knowledge of unit testing with PyTest

Patching is the process of temporarily replacing one function with another during testing. We use patching when we want to test code that relies on data from some external data source e.g. API, database, server etc. One of the main criteria for a successful unit test is that it should have **no dependencies** i.e. no reliance on an external data source to be up to date and clean. This is because our unit tests should be testing if our **code** is working as intended, which we cannot do confidently if we introduce uncertainty by using an external data source.

To demonstrate patching we'll write a function which identifies the NBA league leaders in a chosen scoring metric (points, assists or rebounds) by pulling data from the https://stats.nba.com/ API. This is a good use case because there are no logins or API keys required to use it.

## Connecting to the external data source

The API endpoint we want is the https://stats.nba.com/stats/leagueleaders (this could just as easily be a local/work database). First we set the parameters and headers which are mandatory to connect to the API. We'll use them in the class we define below

# Functions we want to test
We define a class with a method which connects to the API and another which filters the data to get the top players for our chosen metric

- The init function contains our  parameters, headers and endpoint as attributes.
- The *extract_api_data* method extracts data by connecting to the API returning a dataframe
- Finally **the function we want to test** `top_players_by_stat` which returns the top *n* players for a chosen metric


```python
# contents of nba.py

import requests
import json
import pandas as pd

class NBAStatChecker:
    
    def __init__(self):
        
        self.url_base = f'https://stats.nba.com/stats/leagueleaders'
        
        self.parameters = {
                            'LeagueID': '00',
                            'PerMode': 'Totals',
                            'Scope': 'S',
                            'Season': '2019-20',
                            'SeasonType': 'Regular Season',
                            'StatCategory': 'PTS',
                            'ActiveFlag':''
                            }
        self.headers = {
                        'Host': 'stats.nba.com',
                        'Connection': 'keep-alive',
                        'Accept': 'application/json, text/plain, */*',
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
                        'Referer': 'https://stats.nba.com/',
                        "x-nba-stats-origin": "stats",
                        "x-nba-stats-token": "true",
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Accept-Language': 'en-US,en;q=0.9'
                        }

    # Connects to API    
    def extract_api_data(self):
        
        # Get response from API
        response = requests.get(self.url_base, params=self.parameters, headers=self.headers)
        
        # Load response content into a json
        content = json.loads(response.content)

        # Convert to a dataframe
        headers = content['resultSet']['headers']
        rows = content['resultSet']['rowSet']
        df = pd.DataFrame(rows, columns=headers)[['PLAYER', 'PTS', 'AST', 'REB']]

        return df
    
    # Method we want to test
    def top_players_by_metric(self, n_top, metric):
        
        df = self.extract_api_data()
        
        return df.sort_values(metric, ascending=False).head(n_top).reset_index(drop=True)       
```

Here is a snapshot of the output from the `extract_api_data` method. 


```python
nba = NBAStatChecker()

api_data = nba.extract_api_data()

print('Data shape :', api_data.shape)
api_data.head()
```

    Data shape : (529, 4)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PLAYER</th>
      <th>PTS</th>
      <th>AST</th>
      <th>REB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>James Harden</td>
      <td>2335</td>
      <td>512</td>
      <td>446</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Damian Lillard</td>
      <td>1978</td>
      <td>530</td>
      <td>284</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Devin Booker</td>
      <td>1863</td>
      <td>456</td>
      <td>297</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Giannis Antetokounmpo</td>
      <td>1857</td>
      <td>354</td>
      <td>856</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Trae Young</td>
      <td>1778</td>
      <td>560</td>
      <td>255</td>
    </tr>
  </tbody>
</table>
</div>



And here is the output from `top_players_by_metric` where *n=3* and *metric=AST* (assists)

(We'll put this in a csv called *top_player_by_metric.csv* for our next step)


```python
leaders = nba.top_players_by_metric(n_top=3, metric='AST')
leaders.to_csv('top_player_by_metric.csv', index=False)

leaders
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PLAYER</th>
      <th>PTS</th>
      <th>AST</th>
      <th>REB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LeBron James</td>
      <td>1698</td>
      <td>684</td>
      <td>525</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ricky Rubio</td>
      <td>847</td>
      <td>570</td>
      <td>304</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Trae Young</td>
      <td>1778</td>
      <td>560</td>
      <td>255</td>
    </tr>
  </tbody>
</table>
</div>



## Testing
Now we can write a test for our `top_players_by_metric` method. We'll put this in a separate `test_nba.py` file and import our NBAStatChecker class.

This will test will obviously pass as we are just just running the same code as above and making an assertion. The important part is that to produce `output` we need to connect to our API.


```python
# contents of test_nba.py

from nba import NBAStatChecker
import pandas as pd

def test_top_player_by_metric():
    
    output = NBAStatChecker().top_players_by_metric(n_top=3, metric='AST')
    
    expected_output = pd.read_csv('../projects/unit_test_patching/top_player_by_metric.csv')
    
    pd.testing.assert_frame_equal(output, expected_output)
    
```

## Patching
We are now ready to patch so we can run our test without connecting to the API. To do this we want to **replace the data extracted using API with a small dataframe of our own**. This ensures the data is correct and removes dependency on the external data source

Below is the dataframe which we will use to replace the API data from `extract_api_data()` . We'll also create our expected output csv of the top 3 assist players as we did previously.


```python
df_patch = pd.DataFrame(
    {'PLAYER': {0: 'Jonny', 1: 'Genti', 2: 'Steven', 3: 'Rob', 4: 'Ghis', 5: 'Liam', 6:'Wale'},
     'PTS': {0: 30, 1: 14, 2: 10, 3: 15, 4: 20, 5: 8, 6: 12},
     'AST': {0: 12, 1: 15, 2: 10, 3: 5, 4: 0, 5: 11, 6: 10},
     'REB': {0: 10, 1: 5, 2: 10, 3: 15, 4: 0, 5: 15, 6: 4}}
    )

df_patch.to_csv('extract_api_data_patch.csv', index=False)

print(df_patch)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PLAYER</th>
      <th>PTS</th>
      <th>AST</th>
      <th>REB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jonny</td>
      <td>30</td>
      <td>12</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Genti</td>
      <td>14</td>
      <td>15</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Steven</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rob</td>
      <td>15</td>
      <td>5</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ghis</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Liam</td>
      <td>8</td>
      <td>11</td>
      <td>15</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Wale</td>
      <td>12</td>
      <td>10</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



To patch we decorate our test function with the `unittest.mock.patch` decorator. This uses the following format:

`@patch.object(<class>, <method-to-patch>, <function-to-patch-with>)`

Note that we must patch a **function** with another **function** we cannot patch a function with a dataframe directly. So first we need to define a function returns another function to load our csv.


```python
def patch_function(file):
    
    def load_csv_to_patch(*args):
        return pd.read_csv(file)

    return load_csv_to_patch
```

Now we can decorate our test function with the `patch.object` decorator. Wherever `extract_api_data` is found it is replaced by `patch_function`. If we have multiple methods to patch we can stack decorators


```python
# contents of test_nba.py

from nba import NBAStatChecker, patch_function
import pandas as pd
from unittest.mock import patch


@patch.object(NBAStatChecker, 'extract_api_data', patch_function("extract_api_data_patch.csv"))
def test_top_player_by_metric():
    
    output = NBAStatChecker().top_players_by_metric(n_top=3, metric='AST')

    expected_output = pd.read_csv('top_player_by_metric_2.csv')

    pd.testing.assert_frame_equal(output, expected_output)
```

Now if we run pytest again our test passes without the need for connecting to the API or amending our source code. Great!


```python
!pytest -v
```

    ============================= test session starts =============================
    platform win32 -- Python 3.6.5, pytest-6.1.1, py-1.9.0, pluggy-0.13.1 -- C:\Users\jonat\Anaconda3\python.exe
    cachedir: .pytest_cache
    rootdir: C:\Users\jonat\Documents\Python Scripts\jonathanfletcher4.github.io\projects\unit_test_patching
    collecting ... collected 1 item
    
    test_nba.py::test_top_player_by_metric PASSED                            [100%]
    
    ============================== 1 passed in 0.81s ==============================
    

# Conclusion
In this post we've gone through how to use patching to test functions which connect to external data sources

Thanks for reading!
