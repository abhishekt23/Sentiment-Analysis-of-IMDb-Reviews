# Load necessary packages

library(tidyverse)
library(text2vec)
library(glmnet)
library(caret)
library(keras)
library(stopwords)

set.seed(123) # For reproducibility

#------------------------------------------
# 1. Load and Inspect the Data
#------------------------------------------
imdb <- read_csv("IMDB Dataset.csv")

# Check structure
glimpse(imdb)

# The dataset has columns: "review", "sentiment"
# sentiment is "positive" or "negative".
# Convert sentiment to a factor.
imdb$sentiment <- factor(imdb$sentiment, levels = c("negative", "positive"))
loss_history <- c()


#------------------------------------------
# 2. Split into Training and Test Sets
#------------------------------------------
train_index <- createDataPartition(imdb$sentiment, p=0.8, list=FALSE)
train_data <- imdb[train_index, ]
test_data  <- imdb[-train_index, ]

#------------------------------------------
# 3. Preprocessing and Tokenization
#------------------------------------------
# text2vec recommends using an iterator over tokens.
# We'll define a custom function for tokenization and preprocessing.

prep_fun <- tolower
tok_fun <- word_tokenizer

# Remove HTML tags and punctuation
train_data$review <- gsub("<.*?>", "", train_data$review)
test_data$review <- gsub("<.*?>", "", test_data$review)

# Define stopwords (you can use stopwords from a package if desired)
stopwords <- c(stopwords::stopwords("en"), "br") # "br" often appears as artifact in IMDB data

# Tokenizing
it_train <- itoken(train_data$review, 
                   preprocessor = prep_fun, 
                   tokenizer = tok_fun,
                   progressbar = FALSE)

it_test <- itoken(test_data$review, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun,
                  progressbar = FALSE)

#------------------------------------------
# 4. Create Vocabulary and DTM
#------------------------------------------
# Create a vocabulary
vocab <- create_vocabulary(it_train, stopwords = stopwords) %>%
  prune_vocabulary(term_count_min = 5) # prune words that occur < 5 times

# Use a hash vectorizer or vocab_vectorizer
vectorizer <- vocab_vectorizer(vocab)

dtm_train <- create_dtm(it_train, vectorizer)
dtm_test <- create_dtm(it_test, vectorizer)

#------------------------------------------
# 5. Train a glmnet Model
#------------------------------------------
y_train <- train_data$sentiment
y_test <- test_data$sentiment

# glmnet requires numerical targets for classification; 
# we can encode factor levels as 0/1.
y_train_num <- ifelse(y_train == "positive", 1, 0)

# Perform cross-validation with glmnet
cv_fit <- cv.glmnet(x = dtm_train, y = y_train_num, 
                    family = "binomial", 
                    alpha = 1, # LASSO penalty; you can try alpha=0.5 or 0
                    type.measure = "class")

best_lambda <- cv_fit$lambda.min

# Predict on test set
pred_probs <- predict(cv_fit, dtm_test, s = best_lambda, type = "response")
pred_class <- ifelse(pred_probs > 0.5, "positive", "negative") %>% factor(levels=c("negative","positive"))

conf_mat <- confusionMatrix(pred_class, y_test)
conf_mat

# Evaluate accuracy
accuracy_glmnet <- conf_mat$overall["Accuracy"]
cat("GLMNET accuracy:", accuracy_glmnet, "\n")

#------------------------------------------
# 6. Simple Neural Network Model Using torch (Updated)
#------------------------------------------

library(text2vec)
library(tidyverse)
library(caret)
library(torch)

# Assuming you have train_data, test_data, and accuracy_glmnet from previous steps.

# Create a smaller vocabulary to keep dimensions manageable
vocab_small <- create_vocabulary(itoken(train_data$review, tolower, word_tokenizer),
                                 stopwords = stopwords::stopwords("en")) %>%
  prune_vocabulary(term_count_min = 50, doc_proportion_max = 0.4, vocab_term_max = 1000)

vectorizer_small <- vocab_vectorizer(vocab_small)
dtm_train_small <- create_dtm(itoken(train_data$review, tolower, word_tokenizer), vectorizer_small)
dtm_test_small  <- create_dtm(itoken(test_data$review, tolower, word_tokenizer), vectorizer_small)

x_train_mat <- as.matrix(dtm_train_small)
x_test_mat  <- as.matrix(dtm_test_small)

# Apply PCA to reduce the feature space further
pca_model <- prcomp(x_train_mat, center = TRUE, scale. = TRUE)
num_pcs <- 100
x_train_pca <- pca_model$x[, 1:num_pcs]
x_test_pca  <- predict(pca_model, x_test_mat)[, 1:num_pcs]

y_train <- factor(train_data$sentiment, levels = c("negative", "positive"))
y_test  <- factor(test_data$sentiment, levels = c("negative", "positive"))

# Convert targets to numeric and shift by 1 so that negative=1, positive=2
y_train_num <- as.integer(y_train == "positive") + 1
y_test_num  <- as.integer(y_test == "positive") + 1

# Remove any rows with NA
if (anyNA(x_train_pca) || anyNA(y_train_num)) {
  complete_idx <- complete.cases(x_train_pca, y_train_num)
  x_train_pca <- x_train_pca[complete_idx, ]
  y_train_num <- y_train_num[complete_idx]
}

# Convert data to torch tensors
x_train_tensor <- torch_tensor(x_train_pca, dtype = torch_float())
y_train_tensor <- torch_tensor(y_train_num, dtype = torch_long())

x_test_tensor <- torch_tensor(x_test_pca, dtype = torch_float())
y_test_tensor <- torch_tensor(y_test_num, dtype = torch_long())

# Define a simple feed-forward neural network
net <- nn_module(
  initialize = function(input_size, hidden_size, output_size) {
    self$fc1 <- nn_linear(input_size, hidden_size)
    self$fc2 <- nn_linear(hidden_size, output_size)
  },
  forward = function(x) {
    x %>% self$fc1() %>% nnf_relu() %>% self$fc2()
  }
)

model <- net(input_size = num_pcs, hidden_size = 50, output_size = 2)

# Define loss and optimizer
criterion <- nn_cross_entropy_loss()
optimizer <- optim_adam(model$parameters, lr = 0.001)

# Training loop
epochs <- 10
batch_size <- 64

num_samples <- nrow(x_train_pca)
num_batches <- ceiling(num_samples / batch_size)

for (epoch in 1:epochs) {
  model$train()
  shuffle_idx <- sample(num_samples)
  x_train_shuffled <- x_train_tensor[shuffle_idx,]
  y_train_shuffled <- y_train_tensor[shuffle_idx]
  
  total_loss <- 0
  
  for (b in 1:num_batches) {
    start_idx <- (b-1)*batch_size + 1
    end_idx <- min(b*batch_size, num_samples)
    
    x_batch <- x_train_shuffled[start_idx:end_idx,]
    y_batch <- y_train_shuffled[start_idx:end_idx]
    
    optimizer$zero_grad()
    output <- model(x_batch)
    loss <- criterion(output, y_batch)
    loss$backward()
    optimizer$step()
    
    total_loss <- total_loss + loss$item()
  }
  
  loss_history <- c(loss_history, total_loss / num_batches)
  
  
  cat("Epoch:", epoch, "Loss:", total_loss / num_batches, "\n")
}

# Evaluation
model$eval()
with_no_grad({
  output_test <- model(x_test_tensor)
  preds <- output_test$argmax(dim=2)$to(dtype = torch_int())
})

# Convert preds to R factor
predicted_classes <- ifelse(as_array(preds) == 2, "positive", "negative")
predicted_classes <- factor(predicted_classes, levels = c("negative","positive"))

conf_mat_nn <- confusionMatrix(predicted_classes, y_test)
conf_mat_nn
accuracy_nn <- conf_mat_nn$overall["Accuracy"]
cat("Torch NN accuracy:", accuracy_nn, "\n")

# Compare with GLMNET from before
cat("GLMNET Accuracy:", accuracy_glmnet, "\n")
cat("NN Accuracy (Torch):", accuracy_nn, "\n")





install.packages(c("ggplot2", "wordcloud", "RColorBrewer", "pheatmap"))
library(ggplot2)
ggplot(imdb, aes(x = sentiment, fill = sentiment)) +
  geom_bar() +
  labs(title = "Sentiment Distribution in IMDB Dataset",
       x = "Sentiment",
       y = "Count") +
  theme_minimal()

View(imdb)

# Extract and plot coefficients
coef_sparse <- coef(cv_fit, s = best_lambda)
print(best_lambda)
coef_df <- as.data.frame(as.matrix(coef_sparse))

# Rename columns for clarity
colnames(coef_df) <- "Coefficient"
coef_df$Feature <- rownames(coef_df)

# Filter top 20 features by absolute coefficient value
coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), ][1:20, ]

library(ggplot2)

ggplot(coef_df, aes(x = reorder(Feature, abs(Coefficient)), y = Coefficient)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Top 20 Important Features (GLMNET)",
       x = "Feature",
       y = "Coefficient Value") +
  theme_minimal()

# Store loss during training in a vector

# During training loop (inside your training code):
# Append total_loss / num_batches to loss_history at the end of each epoch

# After training:
plot(1:length(loss_history), loss_history, type = "b", col = "blue",
     xlab = "Epoch", ylab = "Loss",
     main = "Neural Network Training Loss")



df <- read.csv("datacollection.csv", 
               skip = 3,  # Skip the header rows
               nrows = 90,  # Limit to first 90 rows
               col.names = c("Face_Num", "Age", "Gender", "Race", "Det_Age", 
                             "Age_Conf", "Det_Gender", "Gender_Conf", 
                             "Det_Race", "Race_Conf", "Quality", "Notes"))

# Group by Race and calculate summary statistics
library(dplyr)
race_data <- df %>%
  group_by(Race) %>%
  summarise(
    Mean_Quality = mean(Quality, na.rm = TRUE),
    Count = n()
  )

# Create the plot
library(ggplot2)
p <- ggplot(race_data, aes(x = Race, y = Mean_Quality, fill = Race)) +
  geom_bar(stat = "identity", color = "black", width = 0.7) +
  geom_text(aes(label = sprintf("%.3f\n(n=%d)", Mean_Quality, Count)), 
            position = position_dodge(width = 0.9), 
            vjust = -0.5) +
  labs(title = "Mean Quality of Result by Race/Ethnicity (First 90 Entries)",
       x = "Race/Ethnicity",
       y = "Mean Quality of Result") +
  theme_minimal() +
  scale_fill_brewer(palette = "Pastel1") +
  theme(legend.position = "none")

p

# Save the plot
ggsave("race_quality_barplot_90entries.png", p, width = 8, height = 6)

# Print summary for reference
print(race_data)




age_data <- df %>%
  group_by(Age) %>%
  summarise(
    Mean_Quality = mean(Quality, na.rm = TRUE),
    Count = n()
  ) %>%
  # Sort by count to ensure more representative categories are prominent
  arrange(desc(Count))

# Create the plot
p <- ggplot(age_data, aes(x = Age, y = Mean_Quality, fill = Age)) +
  geom_bar(stat = "identity", color = "black", width = 0.7) +
  geom_text(aes(label = sprintf("%.3f\n(n=%d)", Mean_Quality, Count)), 
            position = position_dodge(width = 0.9), 
            vjust = -0.5) +
  labs(title = "Mean Quality of Result by Age (First 90 Entries)",
       x = "Age Category",
       y = "Mean Quality of Result") +
  theme_minimal() +
  scale_fill_brewer(palette = "Pastel1") +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1))

p

# Save the plot
ggsave("age_quality_barplot_90entries.png", p, width = 10, height = 6)

# Print summary for reference
print(age_data)




