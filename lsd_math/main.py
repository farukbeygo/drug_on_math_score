import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("lsd_math_score_data.csv")

time_delay = (data["Time_Delay_in_Minutes"]).values.reshape(-1, 1)
lsd_rate = data["LSD_ppm"]
test_score = (data["Avg_Math_Test_Score"] / 10).values.reshape(-1, 1)

regression = LinearRegression()
regression.fit(time_delay, test_score)

plt.style.use("fivethirtyeight")
plt.plot(time_delay, regression.predict(time_delay), color="red")
plt.plot(time_delay, lsd_rate, alpha=0.6)
plt.scatter(time_delay, test_score, alpha=0.6, color="black")

plt.xlabel('time delay')
plt.ylabel('performance')
plt.title('lsd on mathematics test score')

plt.savefig('linear_regression.png')
plt.show()