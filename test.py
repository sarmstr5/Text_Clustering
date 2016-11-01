scorer = make_scorer(silhouette_score, metric='cosine')
score = cross_val_score(svd_model, x, scoring=scorer, cv=folds)
GridSearchCV(TruncatedSVD(), param_grid)