Difference: Full dataset vs current approach
Current approach (what you have now)
Stores only summaries/insights in session
Shows top 10 customers, top 10 high-risk customers
Displays aggregated statistics (mean, median, counts)
Limited data in session to avoid header size errors
Full dataset approach (if we implement file storage)
Stores complete processed DataFrame
Can show all customers (not just top 10)
Can display full data tables
Can export complete results
Can filter/search all data
Results difference
Model performance (no difference)
Same RMSE, MAE, R² scores
Same best model selection
Same predictions for all rows
Models train on the full dataset regardless
Business insights (display difference only)
Insight	Current (Limited)	Full Dataset
Customer Segmentation	Shows 3 segments with sizes	Shows all customers with their segment assignments
CLV Prediction	Top 10 high-value customers	All customers with CLV values
Churn Prediction	Top 10 high-risk customers	All customers with risk levels
Market Basket	Top 5 combinations	All association rules found
Sales Forecasting	10 periods shown	All periods available
What changes
Display: current shows summaries; full shows all data
Functionality: current is limited; full supports filtering, search, export
Accuracy: no difference — models use the full dataset
What stays the same
Model training uses the full dataset
Model accuracy/metrics are identical
Predictions are for all rows
Business insights calculations use all data
Summary
Results/accuracy: no difference — models use the full dataset
Display: current shows summaries; full shows all data
Functionality: current is limited; full supports more features
The current approach calculates everything on the full dataset but only displays summaries. The full dataset approach would let you view, filter, and export all results.
Should I implement the file-based storage so you can access and display all the data?