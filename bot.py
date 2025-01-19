import indicator as ind
import model as md

ind.indicator_calculator("BTC-USD")


trinedModel = md.train_model('training_Data.csv')
