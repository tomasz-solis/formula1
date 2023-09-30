# Formula 1 predictor

## Problems statement
1. Which tracks are similar in terms of its characteristics?
- High/medium/low speed corners
- Full throttle rate
- Altitude
- Weather
2. Which drivers will perform well in the next race?
- Predicting the probability of winning => Classification
- Predicting the ranking => Ranking
3. Which teams will perform well in the next race?
- Predicting the probability of winning => Classification
- Predicting the ranking => Ranking
4. Red flag likelihood in every stage of the race - based on historical races (last X years)
5. Safety Car likelihood in every stage of the race - based on historical races (last X years)

Problems will be solved gradually as answers to the next question often requires the info from the previous one. 

I will not be considering tyre load as it is car specific and requires detailed knowledge of all cars (weight, weight distribution, drag, aerodynamic centre of pressure and much more). All models will be generalizing the performance based on session performance to minimise the impact of variables introduced by teams updates or setup. 

Only pre-Qualifying sessions (Free Practice sessions) will be considered as Qualifying is one of the sessions I would like to predict. 
- FP1/FP2/FP3 for conventional race weekends
- FP1 for sprint race weekends
