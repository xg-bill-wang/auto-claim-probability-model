set.seed(20260820)

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- if (length(script_arg) > 0) sub("^--file=", "", script_arg[1]) else "R/claim_probability_models.R"
root_dir <- normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE)
target <- "claim_status"

resolve_data_path <- function() {
  env_path <- Sys.getenv("INSURANCE_CLAIMS_CSV")
  candidates <- c(
    env_path,
    file.path(root_dir, "data", "raw", "Insurance claims data.csv"),
    "/Users/personal/Downloads/Insurance claims data.csv"
  )
  candidates <- candidates[nzchar(candidates)]
  for (path in candidates) {
    if (file.exists(path)) {
      return(path)
    }
  }
  stop("Could not find the claims CSV. Place it at data/raw/Insurance claims data.csv or set INSURANCE_CLAIMS_CSV.")
}

stratified_split <- function(y, test_size = 0.25) {
  train_idx <- integer(0)
  test_idx <- integer(0)
  for (value in sort(unique(y))) {
    idx <- which(y == value)
    idx <- sample(idx)
    n_test <- round(length(idx) * test_size)
    test_idx <- c(test_idx, idx[seq_len(n_test)])
    train_idx <- c(train_idx, idx[(n_test + 1):length(idx)])
  }
  list(train = sample(train_idx), test = sample(test_idx))
}

auc_score <- function(y_true, scores) {
  n_pos <- sum(y_true == 1)
  n_neg <- sum(y_true == 0)
  if (n_pos == 0 || n_neg == 0) {
    return(NA_real_)
  }
  ranks <- rank(scores, ties.method = "average")
  (sum(ranks[y_true == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

classification_metrics <- function(y_true, prob, threshold) {
  pred <- as.integer(prob >= threshold)
  tp <- sum(pred == 1 & y_true == 1)
  fp <- sum(pred == 1 & y_true == 0)
  tn <- sum(pred == 0 & y_true == 0)
  fn <- sum(pred == 0 & y_true == 1)
  precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
  recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
  f1 <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
  data.frame(
    accuracy = (tp + tn) / length(y_true),
    precision = precision,
    recall = recall,
    f1 = f1,
    true_positive = tp,
    false_positive = fp,
    true_negative = tn,
    false_negative = fn
  )
}

top_decile_lift <- function(y_true, prob) {
  cutoff <- quantile(prob, 0.90, names = FALSE, type = 7)
  mean(y_true[prob >= cutoff]) / mean(y_true)
}

evaluate_model <- function(model_name, y_true, raw_scores, threshold) {
  prob <- pmin(pmax(raw_scores, 0), 1)
  metrics <- classification_metrics(y_true, prob, threshold)
  data.frame(
    model = model_name,
    roc_auc = auc_score(y_true, raw_scores),
    accuracy = metrics$accuracy,
    precision = metrics$precision,
    recall = metrics$recall,
    f1 = metrics$f1,
    brier_score = mean((prob - y_true)^2),
    top_decile_lift = top_decile_lift(y_true, prob),
    mean_predicted_probability = mean(prob),
    threshold = threshold,
    share_raw_outside_0_1 = mean(raw_scores < 0 | raw_scores > 1),
    raw_prediction_min = min(raw_scores),
    raw_prediction_max = max(raw_scores),
    true_positive = metrics$true_positive,
    false_positive = metrics$false_positive,
    true_negative = metrics$true_negative,
    false_negative = metrics$false_negative
  )
}

write_calibration <- function(y_true, scores, model_name) {
  prob <- pmin(pmax(scores, 0), 1)
  ord <- order(prob)
  decile <- integer(length(prob))
  decile[ord] <- ceiling(seq_along(prob) / length(prob) * 10)
  decile <- pmin(pmax(decile, 1), 10)
  frame <- data.frame(decile = decile, actual = y_true, predicted = prob)
  rows <- lapply(sort(unique(decile)), function(d) {
    subset <- frame[frame$decile == d, ]
    data.frame(
      decile = d,
      policies = nrow(subset),
      predicted_claim_rate = mean(subset$predicted),
      actual_claim_rate = mean(subset$actual),
      model = model_name
    )
  })
  do.call(rbind, rows)
}

plot_model_performance <- function(metrics, path) {
  chart_metrics <- c("roc_auc", "precision", "recall", "f1")
  labels <- c("ROC-AUC", "Precision", "Recall", "F1")
  colors <- c("#235789", "#c1292e", "#f1d302")
  width <- 920
  height <- 430
  left <- 80
  right <- 40
  top <- 70
  bottom <- 90
  plot_w <- width - left - right
  plot_h <- height - top - bottom
  group_w <- plot_w / length(chart_metrics)
  bar_w <- group_w / (nrow(metrics) + 1)
  parts <- c(
    sprintf('<svg xmlns="http://www.w3.org/2000/svg" width="%s" height="%s" viewBox="0 0 %s %s">', width, height, width, height),
    '<rect width="100%" height="100%" fill="#ffffff"/>',
    '<text x="40" y="38" font-family="Arial" font-size="24" font-weight="700" fill="#111827">R model performance on holdout policies</text>',
    '<text x="40" y="62" font-family="Arial" font-size="13" fill="#4b5563">Threshold set at training base claim rate; higher is better.</text>'
  )
  for (tick in seq(0, 1, by = 0.2)) {
    y <- top + plot_h * (1 - tick)
    parts <- c(parts,
      sprintf('<line x1="%s" y1="%.1f" x2="%s" y2="%.1f" stroke="#e5e7eb"/>', left, y, width - right, y),
      sprintf('<text x="48" y="%.1f" font-family="Arial" font-size="12" fill="#6b7280">%.1f</text>', y + 4, tick)
    )
  }
  for (i in seq_along(chart_metrics)) {
    x0 <- left + (i - 1) * group_w
    for (j in seq_len(nrow(metrics))) {
      value <- metrics[j, chart_metrics[i]]
      bar_h <- value * plot_h
      x <- x0 + (j - 0.5) * bar_w
      y <- top + plot_h - bar_h
      parts <- c(parts, sprintf('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s"/>', x, y, bar_w * 0.78, bar_h, colors[j]))
    }
    parts <- c(parts, sprintf('<text x="%.1f" y="%s" text-anchor="middle" font-family="Arial" font-size="13" fill="#111827">%s</text>', x0 + group_w / 2, height - 45, labels[i]))
  }
  for (j in seq_len(nrow(metrics))) {
    x <- 80 + (j - 1) * 250
    parts <- c(parts,
      sprintf('<rect x="%s" y="%s" width="14" height="14" fill="%s"/>', x, height - 26, colors[j]),
      sprintf('<text x="%s" y="%s" font-family="Arial" font-size="13" fill="#111827">%s</text>', x + 22, height - 14, metrics$model[j])
    )
  }
  parts <- c(parts, "</svg>")
  writeLines(parts, path)
}

plot_logistic_calibration <- function(calibration, path) {
  logistic <- calibration[calibration$model == "Logistic regression", ]
  width <- 820
  height <- 460
  left <- 72
  right <- 40
  top <- 70
  bottom <- 70
  plot_w <- width - left - right
  plot_h <- height - top - bottom
  ymax <- max(logistic$predicted_claim_rate, logistic$actual_claim_rate, 0.10)
  point <- function(rate, decile) {
    c(
      left + (decile - 1) / 9 * plot_w,
      top + plot_h * (1 - rate / ymax)
    )
  }
  actual_points <- paste(sapply(seq_len(nrow(logistic)), function(i) paste(round(point(logistic$actual_claim_rate[i], logistic$decile[i]), 1), collapse = ",")), collapse = " ")
  predicted_points <- paste(sapply(seq_len(nrow(logistic)), function(i) paste(round(point(logistic$predicted_claim_rate[i], logistic$decile[i]), 1), collapse = ",")), collapse = " ")
  parts <- c(
    sprintf('<svg xmlns="http://www.w3.org/2000/svg" width="%s" height="%s" viewBox="0 0 %s %s">', width, height, width, height),
    '<rect width="100%" height="100%" fill="#ffffff"/>',
    '<text x="38" y="38" font-family="Arial" font-size="24" font-weight="700" fill="#111827">R logistic regression calibration by risk decile</text>',
    '<text x="38" y="61" font-family="Arial" font-size="13" fill="#4b5563">Predicted probabilities should move with observed claim rates.</text>'
  )
  for (tick in seq(0, ymax, length.out = 6)) {
    y <- top + plot_h * (1 - tick / ymax)
    parts <- c(parts,
      sprintf('<line x1="%s" y1="%.1f" x2="%s" y2="%.1f" stroke="#e5e7eb"/>', left, y, width - right, y),
      sprintf('<text x="30" y="%.1f" font-family="Arial" font-size="12" fill="#6b7280">%.1f%%</text>', y + 4, tick * 100)
    )
  }
  parts <- c(parts,
    sprintf('<polyline fill="none" stroke="#235789" stroke-width="3" points="%s"/>', predicted_points),
    sprintf('<polyline fill="none" stroke="#c1292e" stroke-width="3" points="%s"/>', actual_points)
  )
  for (i in seq_len(nrow(logistic))) {
    p1 <- point(logistic$predicted_claim_rate[i], logistic$decile[i])
    p2 <- point(logistic$actual_claim_rate[i], logistic$decile[i])
    parts <- c(parts,
      sprintf('<circle cx="%.1f" cy="%.1f" r="4" fill="#235789"/>', p1[1], p1[2]),
      sprintf('<circle cx="%.1f" cy="%.1f" r="4" fill="#c1292e"/>', p2[1], p2[2]),
      sprintf('<text x="%.1f" y="%s" text-anchor="middle" font-family="Arial" font-size="12" fill="#374151">%s</text>', left + (logistic$decile[i] - 1) / 9 * plot_w, height - 42, logistic$decile[i])
    )
  }
  parts <- c(parts,
    '<text x="358" y="430" font-family="Arial" font-size="13" fill="#374151">Risk decile, low to high</text>',
    '<rect x="600" y="84" width="14" height="14" fill="#235789"/><text x="622" y="96" font-family="Arial" font-size="13" fill="#111827">Predicted</text>',
    '<rect x="600" y="106" width="14" height="14" fill="#c1292e"/><text x="622" y="118" font-family="Arial" font-size="13" fill="#111827">Observed</text>',
    "</svg>"
  )
  writeLines(parts, path)
}

main <- function() {
  data_path <- resolve_data_path()
  claims <- read.csv(data_path, stringsAsFactors = TRUE)
  if (!target %in% names(claims)) {
    stop(paste("Expected target column", target))
  }
  claims[[target]] <- as.integer(claims[[target]])

  split <- stratified_split(claims[[target]])
  train <- claims[split$train, ]
  test <- claims[split$test, ]
  train_model <- train[, setdiff(names(train), "policy_id")]
  test_model <- test[, setdiff(names(test), "policy_id")]
  threshold <- mean(train[[target]])

  simple_lm <- lm(claim_status ~ subscription_length, data = train_model)
  multi_lpm <- lm(claim_status ~ ., data = train_model)
  logit_glm <- glm(claim_status ~ ., data = train_model, family = binomial(link = "logit"))

  simple_scores <- as.numeric(predict(simple_lm, newdata = test_model))
  multi_scores <- as.numeric(predict(multi_lpm, newdata = test_model))
  logit_scores <- as.numeric(predict(logit_glm, newdata = test_model, type = "response"))
  y_test <- test_model[[target]]

  metrics <- rbind(
    evaluate_model("Simple linear baseline", y_test, simple_scores, threshold),
    evaluate_model("Multiple linear probability", y_test, multi_scores, threshold),
    evaluate_model("Logistic regression", y_test, logit_scores, threshold)
  )

  calibration <- rbind(
    write_calibration(y_test, simple_scores, "Simple linear baseline"),
    write_calibration(y_test, multi_scores, "Multiple linear probability"),
    write_calibration(y_test, logit_scores, "Logistic regression")
  )

  coefficients <- summary(logit_glm)$coefficients
  coefficient_table <- data.frame(
    feature = rownames(coefficients),
    coefficient = coefficients[, "Estimate"],
    p_value = coefficients[, "Pr(>|z|)"],
    row.names = NULL
  )
  coefficient_table$absolute_coefficient <- abs(coefficient_table$coefficient)
  coefficient_table <- coefficient_table[coefficient_table$feature != "(Intercept)", ]
  coefficient_table <- coefficient_table[order(-coefficient_table$absolute_coefficient), ][1:20, ]

  output_dir <- file.path(root_dir, "outputs")
  figure_dir <- file.path(root_dir, "figures")
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(figure_dir, showWarnings = FALSE, recursive = TRUE)

  write.csv(metrics, file.path(output_dir, "r_model_metrics.csv"), row.names = FALSE)
  write.csv(calibration, file.path(output_dir, "r_calibration_by_decile.csv"), row.names = FALSE)
  write.csv(coefficient_table, file.path(output_dir, "r_top_logistic_coefficients.csv"), row.names = FALSE)
  write.csv(
    data.frame(
      item = c("source_path", "rows", "columns", "claim_count", "claim_rate", "train_rows", "test_rows"),
      value = c(data_path, nrow(claims), ncol(claims), sum(claims[[target]]), mean(claims[[target]]), nrow(train), nrow(test))
    ),
    file.path(output_dir, "r_data_summary.csv"),
    row.names = FALSE
  )

  plot_model_performance(metrics, file.path(figure_dir, "r_model_performance.svg"))
  plot_logistic_calibration(calibration, file.path(figure_dir, "r_logistic_calibration.svg"))

  print(paste("Data:", data_path))
  print(paste("Rows:", nrow(claims), "Claim rate:", sprintf("%.2f%%", mean(claims[[target]]) * 100)))
  printable_metrics <- metrics[, c("model", "roc_auc", "accuracy", "precision", "recall", "f1", "brier_score", "top_decile_lift")]
  numeric_cols <- names(printable_metrics)[sapply(printable_metrics, is.numeric)]
  printable_metrics[numeric_cols] <- round(printable_metrics[numeric_cols], 4)
  print(printable_metrics)
  print(paste("Wrote R outputs to", output_dir))
}

main()
