package database

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/brianvoe/gofakeit/v7"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// SeedMLAnalysisLogs creates ML analysis logs for security analysis
func SeedMLAnalysisLogs(ctx context.Context, pool *pgxpool.Pool) error {
	log.Printf("🔍 Seeding ML analysis logs...")

	// Get some HTTP request logs to reference
	var httpLogs []uuid.UUID
	rows, err := pool.Query(ctx, "SELECT id FROM http_request_logs LIMIT $1", ML_ML_ANALYSIS_LOGS_COUNT)
	if err != nil {
		return fmt.Errorf("failed to fetch HTTP logs: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var id uuid.UUID
		if err := rows.Scan(&id); err != nil {
			continue
		}
		httpLogs = append(httpLogs, id)
	}

	if len(httpLogs) == 0 {
		log.Println("No HTTP logs found, skipping ML analysis logs")
		return nil
	}

	logs := make([][]interface{}, 0)
	analysisTypes := []string{"security_classification", "anomaly_detection", "pattern_matching", "ensemble_analysis"}
	modelVersions := []string{"v2.0", "v2.1", "v2.2"}

	for i := 0; i < ML_ML_ANALYSIS_LOGS_COUNT; i++ {
		id := uuid.New()
		requestLogID := httpLogs[gofakeit.IntRange(0, len(httpLogs)-1)]
		analysisType := analysisTypes[gofakeit.IntRange(0, len(analysisTypes)-1)]
		modelVersion := modelVersions[gofakeit.IntRange(0, len(modelVersions)-1)]

		// Generate realistic ML analysis data
		inputFeatures := map[string]interface{}{
			"path_length":         gofakeit.IntRange(5, 200),
			"query_params":        gofakeit.IntRange(0, 10),
			"special_chars":       gofakeit.IntRange(0, 50),
			"entropy":             gofakeit.Float64Range(0.1, 5.0),
			"suspicious_keywords": gofakeit.IntRange(0, 5),
		}

		mlProbabilities := map[string]interface{}{
			"benign": gofakeit.Float64Range(0.1, 0.9),
			"xss":    gofakeit.Float64Range(0.05, 0.4),
			"sqli":   gofakeit.Float64Range(0.05, 0.4),
			"csrf":   gofakeit.Float64Range(0.05, 0.3),
		}

		patternResults := map[string]interface{}{
			"xss_patterns":  gofakeit.IntRange(0, 3),
			"sql_patterns":  gofakeit.IntRange(0, 2),
			"csrf_patterns": gofakeit.IntRange(0, 1),
		}

		ensembleDecision := map[string]interface{}{
			"final_prediction": gofakeit.RandomString([]string{"benign", "xss", "sqli", "csrf"}),
			"confidence":       gofakeit.Float64Range(0.6, 0.98),
			"voting_results":   map[string]string{"nb": "benign", "svm": "xss", "rf": "benign"},
		}

		inputFeaturesJSON, _ := json.Marshal(inputFeatures)
		mlProbabilitiesJSON, _ := json.Marshal(mlProbabilities)
		patternResultsJSON, _ := json.Marshal(patternResults)
		ensembleDecisionJSON, _ := json.Marshal(ensembleDecision)

		processingTime := gofakeit.IntRange(10, 500)
		timestamp := gofakeit.DateRange(time.Now().AddDate(0, 0, -30), time.Now())

		logs = append(logs, []interface{}{
			id,
			requestLogID,
			analysisType,
			modelVersion,
			inputFeaturesJSON,
			mlProbabilitiesJSON,
			patternResultsJSON,
			ensembleDecisionJSON,
			nil, // feature_importance
			nil, // explanation_data
			processingTime,
			timestamp,
		})

		// Insert in batches
		if len(logs) >= 1000 || i == ML_ML_ANALYSIS_LOGS_COUNT-1 {
			_, err := pool.CopyFrom(
				ctx,
				pgx.Identifier{"ml_analysis_logs"},
				[]string{"id", "request_log_id", "analysis_type", "model_version",
					"input_features", "ml_probabilities", "pattern_results", "ensemble_decision",
					"feature_importance", "explanation_data", "processing_time_ms", "timestamp"},
				pgx.CopyFromRows(logs),
			)
			if err != nil {
				return fmt.Errorf("failed to copy ML analysis logs batch: %w", err)
			}
			logs = logs[:0]
		}
	}

	log.Printf("✅ Successfully created %d ML analysis logs", ML_ML_ANALYSIS_LOGS_COUNT)
	return nil
}

// SeedSecurityMetrics creates daily security metrics
func SeedSecurityMetrics(ctx context.Context, pool *pgxpool.Pool) error {
	log.Printf("📊 Seeding security metrics...")

	metrics := make([][]interface{}, 0)

	// Create metrics for last 90 days
	for i := 0; i < 90; i++ {
		id := uuid.New()
		date := time.Now().AddDate(0, 0, -i)

		totalRequests := gofakeit.IntRange(100, 1000)
		maliciousRequests := gofakeit.IntRange(5, 50)

		attackTypeCounts := map[string]interface{}{
			"xss":    gofakeit.IntRange(2, 20),
			"sqli":   gofakeit.IntRange(1, 15),
			"csrf":   gofakeit.IntRange(1, 10),
			"benign": totalRequests - maliciousRequests,
		}

		topAttackSources := map[string]interface{}{
			"192.168.1.100": gofakeit.IntRange(5, 15),
			"10.0.0.50":     gofakeit.IntRange(3, 12),
			"172.16.0.25":   gofakeit.IntRange(2, 8),
		}

		attackTypeCountsJSON, _ := json.Marshal(attackTypeCounts)
		topAttackSourcesJSON, _ := json.Marshal(topAttackSources)

		falsePositiveRate := gofakeit.Float64Range(0.01, 0.1)
		falseNegativeRate := gofakeit.Float64Range(0.01, 0.05)
		averageConfidence := gofakeit.Float64Range(0.8, 0.95)
		modelAccuracy := gofakeit.Float64Range(0.85, 0.98)

		metrics = append(metrics, []interface{}{
			id,
			date.Format("2006-01-02"),
			totalRequests,
			maliciousRequests,
			attackTypeCountsJSON,
			falsePositiveRate,
			falseNegativeRate,
			averageConfidence,
			gofakeit.Float64Range(50.0, 200.0), // average_processing_time
			modelAccuracy,
			topAttackSourcesJSON,
			nil, // top_attack_paths
		})
	}

	_, err := pool.CopyFrom(
		ctx,
		pgx.Identifier{"security_metrics"},
		[]string{"id", "date", "total_requests", "malicious_requests", "attack_type_counts",
			"false_positive_rate", "false_negative_rate", "average_confidence",
			"average_processing_time", "model_accuracy", "top_attack_sources", "top_attack_paths"},
		pgx.CopyFromRows(metrics),
	)

	if err != nil {
		return fmt.Errorf("failed to copy security metrics: %w", err)
	}

	log.Printf("✅ Successfully created %d security metrics entries", len(metrics))
	return nil
}

// SeedSecurityFeedback creates security feedback entries
func SeedSecurityFeedback(ctx context.Context, pool *pgxpool.Pool) error {
	log.Printf("🔄 Seeding security feedback...")

	// Get some HTTP request logs to reference
	var httpLogs []uuid.UUID
	rows, err := pool.Query(ctx, "SELECT id FROM http_request_logs LIMIT $1", ML_SECURITY_FEEDBACK_COUNT*2)
	if err != nil {
		return fmt.Errorf("failed to fetch HTTP logs: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var id uuid.UUID
		if err := rows.Scan(&id); err != nil {
			continue
		}
		httpLogs = append(httpLogs, id)
	}

	if len(httpLogs) == 0 {
		log.Println("No HTTP logs found, skipping security feedback")
		return nil
	}

	feedback := make([][]interface{}, 0)
	feedbackTypes := []string{"false_positive", "false_negative", "correct", "unknown"}
	feedbackSources := []string{"admin", "user", "automated"}
	predictions := []string{"benign", "xss", "sqli", "csrf"}

	for i := 0; i < ML_SECURITY_FEEDBACK_COUNT; i++ {
		id := uuid.New()
		requestLogID := httpLogs[gofakeit.IntRange(0, len(httpLogs)-1)]
		feedbackType := feedbackTypes[gofakeit.IntRange(0, len(feedbackTypes)-1)]
		originalPrediction := predictions[gofakeit.IntRange(0, len(predictions)-1)]
		correctedLabel := predictions[gofakeit.IntRange(0, len(predictions)-1)]
		feedbackSource := feedbackSources[gofakeit.IntRange(0, len(feedbackSources)-1)]

		feedbackReason := gofakeit.Sentence(5)
		confidenceInFeedback := gofakeit.Float64Range(0.7, 1.0)
		createdAt := gofakeit.DateRange(time.Now().AddDate(0, 0, -60), time.Now())

		feedback = append(feedback, []interface{}{
			id,
			requestLogID,
			feedbackType,
			originalPrediction,
			correctedLabel,
			feedbackSource,
			feedbackReason,
			confidenceInFeedback,
			createdAt,
			nil, // created_by
		})
	}

	_, err = pool.CopyFrom(
		ctx,
		pgx.Identifier{"security_feedback"},
		[]string{"id", "request_log_id", "feedback_type", "original_prediction",
			"corrected_label", "feedback_source", "feedback_reason",
			"confidence_in_feedback", "created_at", "created_by"},
		pgx.CopyFromRows(feedback),
	)

	if err != nil {
		return fmt.Errorf("failed to copy security feedback: %w", err)
	}

	log.Printf("✅ Successfully created %d security feedback entries", ML_SECURITY_FEEDBACK_COUNT)
	return nil
}

// SeedAttackMitigation creates attack mitigation strategies
func SeedAttackMitigation(ctx context.Context, pool *pgxpool.Pool) error {
	log.Printf("🛡️ Seeding attack mitigation strategies...")

	mitigations := [][]interface{}{
		{uuid.New(), "xss", "<script>", "HTML Entity Encoding", "html.escape(user_input)", "Validate and sanitize all user inputs", "high", time.Now(), time.Now()},
		{uuid.New(), "xss", "javascript:", "URL Validation", "urllib.parse.urlparse(url).scheme not in ['javascript', 'data']", "Whitelist allowed URL schemes", "medium", time.Now(), time.Now()},
		{uuid.New(), "sqli", "' OR '", "Parameterized Queries", "cursor.execute(\"SELECT * FROM users WHERE id = %s\", (user_id,))", "Use prepared statements", "critical", time.Now(), time.Now()},
		{uuid.New(), "sqli", "UNION SELECT", "Input Validation", "re.sub(r'[^a-zA-Z0-9]', '', user_input)", "Validate input format", "critical", time.Now(), time.Now()},
		{uuid.New(), "csrf", "missing_token", "CSRF Token Validation", "@csrf_protect decorator", "Implement CSRF tokens", "high", time.Now(), time.Now()},
		{uuid.New(), "csrf", "referer_mismatch", "Referer Validation", "if request.headers.get('Referer') != expected_origin", "Validate request origin", "medium", time.Now(), time.Now()},
	}

	_, err := pool.CopyFrom(
		ctx,
		pgx.Identifier{"attack_mitigation"},
		[]string{"id", "attack_type", "attack_pattern", "mitigation_strategy",
			"sanitization_code", "prevention_tips", "severity_level", "created_at", "updated_at"},
		pgx.CopyFromRows(mitigations),
	)

	if err != nil {
		return fmt.Errorf("failed to copy attack mitigation: %w", err)
	}

	log.Printf("✅ Successfully created %d attack mitigation strategies", len(mitigations))
	return nil
}

// SeedModelPerformanceLogs creates model performance evaluation logs
func SeedModelPerformanceLogs(ctx context.Context, pool *pgxpool.Pool) error {
	log.Printf("📈 Seeding model performance logs...")

	logs := make([][]interface{}, 0)
	modelVersions := []string{"v1.0", "v1.5", "v2.0", "v2.1", "v2.2"}
	evaluationTypes := []string{"cross_validation", "holdout", "benchmark", "ablation"}

	for i := 0; i < 50; i++ {
		id := uuid.New()
		modelVersion := modelVersions[gofakeit.IntRange(0, len(modelVersions)-1)]
		evaluationType := evaluationTypes[gofakeit.IntRange(0, len(evaluationTypes)-1)]

		evaluationData := map[string]interface{}{
			"dataset_size": gofakeit.IntRange(1000, 10000),
			"test_split":   gofakeit.Float64Range(0.2, 0.3),
			"features":     gofakeit.IntRange(15, 50),
		}

		accuracy := gofakeit.Float64Range(0.85, 0.98)
		precision := gofakeit.Float64Range(0.80, 0.95)
		recall := gofakeit.Float64Range(0.82, 0.96)
		f1 := 2 * (precision * recall) / (precision + recall)
		rocAuc := gofakeit.Float64Range(0.88, 0.99)

		evaluationDataJSON, _ := json.Marshal(evaluationData)
		timestamp := gofakeit.DateRange(time.Now().AddDate(0, 0, -180), time.Now())

		logs = append(logs, []interface{}{
			id,
			modelVersion,
			evaluationType,
			evaluationDataJSON,
			accuracy,
			precision,
			recall,
			f1,
			rocAuc,
			timestamp,
		})
	}

	_, err := pool.CopyFrom(
		ctx,
		pgx.Identifier{"model_performance_logs"},
		[]string{"id", "model_version", "evaluation_type", "evaluation_data",
			"accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc", "evaluation_timestamp"},
		pgx.CopyFromRows(logs),
	)

	if err != nil {
		return fmt.Errorf("failed to copy model performance logs: %w", err)
	}

	log.Printf("✅ Successfully created %d model performance logs", len(logs))
	return nil
}

// SeedAblationStudyResults creates ablation study results
func SeedAblationStudyResults(ctx context.Context, pool *pgxpool.Pool) error {
	log.Printf("🎯 Seeding ablation study results...")

	studies := [][]interface{}{
		{uuid.New(), "Feature Importance Study", "pattern_matching", 0.92, 0.87, 0.05, 0.15, 5000, time.Now()},
		{uuid.New(), "Feature Importance Study", "ml_classifier", 0.92, 0.78, 0.14, 0.35, 5000, time.Now()},
		{uuid.New(), "Feature Importance Study", "tfidf_features", 0.92, 0.89, 0.03, 0.08, 5000, time.Now()},
		{uuid.New(), "Feature Importance Study", "embeddings", 0.92, 0.85, 0.07, 0.18, 5000, time.Now()},
		{uuid.New(), "Ensemble Components", "naive_bayes", 0.90, 0.83, 0.07, 0.20, 3000, time.Now()},
		{uuid.New(), "Ensemble Components", "svm", 0.90, 0.86, 0.04, 0.12, 3000, time.Now()},
		{uuid.New(), "Ensemble Components", "random_forest", 0.90, 0.88, 0.02, 0.06, 3000, time.Now()},
	}

	_, err := pool.CopyFrom(
		ctx,
		pgx.Identifier{"ablation_study_results"},
		[]string{"id", "study_name", "component_removed", "baseline_accuracy",
			"reduced_accuracy", "performance_impact", "component_importance",
			"test_samples", "study_timestamp"},
		pgx.CopyFromRows(studies),
	)

	if err != nil {
		return fmt.Errorf("failed to copy ablation study results: %w", err)
	}

	log.Printf("✅ Successfully created %d ablation study results", len(studies))
	return nil
}

// SeedVisualizationData creates visualization data for analysis
func SeedVisualizationData(ctx context.Context, pool *pgxpool.Pool) error {
	log.Printf("📊 Seeding visualization data...")

	visualizations := make([][]interface{}, 0)
	vizTypes := []string{"tsne", "umap", "pca"}

	for _, vizType := range vizTypes {
		id := uuid.New()

		// Generate sample 2D coordinates
		numPoints := gofakeit.IntRange(100, 500)
		dataPoints := make([]map[string]interface{}, numPoints)

		for i := 0; i < numPoints; i++ {
			dataPoints[i] = map[string]interface{}{
				"x":     gofakeit.Float64Range(-10.0, 10.0),
				"y":     gofakeit.Float64Range(-10.0, 10.0),
				"label": gofakeit.RandomString([]string{"benign", "xss", "sqli", "csrf"}),
				"id":    gofakeit.UUID(),
			}
		}

		parameters := map[string]interface{}{
			"perplexity":    30,
			"learning_rate": 200,
			"n_components":  2,
			"random_state":  42,
		}

		datasetInfo := map[string]interface{}{
			"total_samples": numPoints,
			"features":      gofakeit.IntRange(20, 50),
			"classes":       4,
			"created_from":  "security_analysis_features",
		}

		dataPointsJSON, _ := json.Marshal(dataPoints)
		parametersJSON, _ := json.Marshal(parameters)
		datasetInfoJSON, _ := json.Marshal(datasetInfo)

		visualizations = append(visualizations, []interface{}{
			id,
			vizType,
			dataPointsJSON,
			parametersJSON,
			datasetInfoJSON,
			time.Now(),
		})
	}

	_, err := pool.CopyFrom(
		ctx,
		pgx.Identifier{"visualization_data"},
		[]string{"id", "visualization_type", "data_points", "parameters", "dataset_info", "created_at"},
		pgx.CopyFromRows(visualizations),
	)

	if err != nil {
		return fmt.Errorf("failed to copy visualization data: %w", err)
	}

	log.Printf("✅ Successfully created %d visualization datasets", len(visualizations))
	return nil
}
