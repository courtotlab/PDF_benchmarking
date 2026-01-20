if (!require(ggplot2, quietly = TRUE)) {
    install.packages("ggplot2")
}
library(ggplot2)
options(stringsAsFactors = FALSE)

if (!require(remotes, quietly = TRUE)) {
    install.packages("remotes")
}
if (!require(yogitools, quietly = TRUE)) {
    remotes::install_github("jweile/yogitools", force = TRUE)
}
library(yogitools)

#dev.off()
library(dplyr)
library(tidyr)

# Read the CSV file with full path
df <- read.csv("/Users/ayu/PDF_benchmarking/graphs/Hospitalfinal.csv")

# Create Input column based on LLM column
df$Input <- ifelse(grepl("image", df$LLM, ignore.case = TRUE), "Image", "Text")

# Fill NA values in Prompt column with "Normal"
df$Prompt[is.na(df$Prompt)] <- "Normal"
df$Prompt[df$Prompt == "None"] <- "zero-shot"

# Drop columns with 'unnamed' in the name (case insensitive)
unnamed_cols <- grep("unnamed", names(df), ignore.case = TRUE)
if (length(unnamed_cols) > 0) {
    df <- df[, -unnamed_cols]
}

# Count values in Input column
table(df$Input)

# Group by LLM, Prompt, and Input, then calculate mean and std of F1score
df_stats <- df %>%
    group_by(LLM, Prompt, Input) %>%
    summarise(
        mean = mean(F1score, na.rm = TRUE),
        std = sd(F1score, na.rm = TRUE),
        .groups = 'drop'
    )

# Clean up LLM names
df_stats$LLM <- gsub("\\*ImageInput\\*", "", df_stats$LLM)
df_stats$LLM <- gsub(" \\(One-shot\\)", "", df_stats$LLM)
df_stats$LLM <- gsub(" \\(Zero-shot\\)", "", df_stats$LLM)

# Define name mapping function
name_mapping <- function(x) {
  case_when(
    grepl("gpt-4.1-mini", x) ~ "GPT-4.1",
        grepl("llama3.170b", x) ~ "Llama3.1",
        grepl("NuExtract:4B", x) ~ "NuExtract",
        grepl("mistral\\(24b\\)", x) ~ "Mistral",
        grepl("gemma327b", x) ~ "Gemma",
        x == "GLiNER" ~ "GliNER",
        grepl("biomed-base-v1.0", x) ~ "BioGliNER",
        TRUE ~ x
    )
}

# Apply name mapping
df_stats$LLM <- name_mapping(df_stats$LLM)

# View the results
print(df_stats)

# Convert df_stats to f1_data format for plotting
f1_data <- df_stats %>%
    rename(model = LLM, f1 = mean) %>%
    mutate(
        prompt = case_when(
            Prompt == "Normal" ~ "zero-shot",
            Prompt == "LTNER/GPT-NER" ~ "one-shot",
            Prompt == "None" ~ "zero-shot",
            TRUE ~ tolower(Prompt)
        ),
        input = tolower(Input)
    ) %>%
    select(model, f1, prompt, input)

combos <- expand.grid(
    list(prompt = c("zero-shot", "one-shot"), input = c("text", "image")), 
    stringsAsFactors = FALSE
)
combos$label <- apply(combos, 1, paste, collapse = ";")

# Get all unique models
mnames <- unique(f1_data$model)

# Create a complete grid to ensure all combinations exist
complete_grid <- expand.grid(
    model = mnames,
    prompt = c("zero-shot", "one-shot"),
    input = c("text", "image"),
    stringsAsFactors = FALSE
)

# Merge with actual data to fill missing combinations with NA
f1_complete <- merge(complete_grid, f1_data, all.x = TRUE)

# Create F1 matrix with consistent dimensions
f1_mat <- matrix(NA, nrow = length(mnames), ncol = nrow(combos))
rownames(f1_mat) <- mnames
colnames(f1_mat) <- combos$label

for (i in seq_along(mnames)) {
    for (j in seq_len(nrow(combos))) {
        combo <- combos[j, ]
        val <- f1_complete$f1[f1_complete$model == mnames[i] & 
                              f1_complete$prompt == combo$prompt & 
                              f1_complete$input == combo$input]
        if (length(val) > 0 && !is.na(val)) {
            f1_mat[i, j] <- val
        }
    }
}

# Create standard error matrix with consistent dimensions  
stderr_mat <- matrix(NA, nrow = length(mnames), ncol = nrow(combos))
rownames(stderr_mat) <- mnames
colnames(stderr_mat) <- combos$label

for (i in seq_along(mnames)) {
    for (j in seq_len(nrow(combos))) {
        combo <- combos[j, ]
        
        # More flexible prompt matching for different model types
        matching_rows <- df_stats[
            df_stats$LLM == mnames[i] & 
            tolower(df_stats$Input) == combo$input,
        ]
        
        # Filter by prompt type
        if (combo$prompt == "zero-shot") {
            matching_rows <- matching_rows[
                matching_rows$Prompt %in% c("Normal", "None", "zero-shot", "", NA) | 
                is.na(matching_rows$Prompt) |
                matching_rows$Prompt == "",
            ]
        } else if (combo$prompt == "one-shot") {
            matching_rows <- matching_rows[
                matching_rows$Prompt == "LTNER/GPT-NER",
            ]
        }
        
        if (nrow(matching_rows) > 0 && !is.na(matching_rows$std[1])) {
            stderr_mat[i, j] <- abs(matching_rows$std[1])  # Force positive error bars
        }
    }
}

# Add debugging output to see what's happening
cat("\nDebugging standard error calculation:\n")
cat("Models in mnames:", paste(mnames, collapse = ", "), "\n")

# Check specific models
for (model_name in c("GliNER", "BioGliNER", "NuExtract")) {
    if (model_name %in% mnames) {
        model_data <- df_stats[df_stats$LLM == model_name, ]
        cat("\n", model_name, " data:\n")
        print(model_data)
        
        # Check stderr_mat for this model
        model_row <- which(mnames == model_name)
        cat("Standard errors for", model_name, ":", stderr_mat[model_row, ], "\n")
    }
}

# Use the stderr_mat as stderr
stderr <- stderr_mat

# Calculate real p-values using statistical tests
# Get raw F1 scores for each condition to perform statistical comparisons
raw_data_for_stats <- df %>%
    mutate(
        prompt_clean = case_when(
            Prompt == "Normal" ~ "zero-shot",
            Prompt == "LTNER/GPT-NER" ~ "one-shot", 
            Prompt == "None" ~ "zero-shot",
            TRUE ~ tolower(Prompt)
        ),
        input_clean = tolower(Input),
        llm_clean = name_mapping(gsub("\\*ImageInput\\*|\\s\\(One-shot\\)|\\s\\(Zero-shot\\)", "", LLM))
    )

# P-value 1: Compare GPT-4.1 zero-shot text vs Gemma on zero-shot image
group1_comparison <- raw_data_for_stats$F1score[
    raw_data_for_stats$llm_clean == "GPT-4.1" & 
    raw_data_for_stats$prompt_clean == "zero-shot" & 
    raw_data_for_stats$input_clean == "text"
]
group2_comparison <- raw_data_for_stats$F1score[
    raw_data_for_stats$llm_clean == "Gemma" & 
    raw_data_for_stats$prompt_clean == "zero-shot" & 
    raw_data_for_stats$input_clean == "image"
]

# Remove NA values
group1_comparison <- group1_comparison[!is.na(group1_comparison)]
group2_comparison <- group2_comparison[!is.na(group2_comparison)]

# Calculate p-value if both groups have data
pval1 <- if (length(group1_comparison) > 0 && length(group2_comparison) > 0) {
    wilcox.test(group1_comparison, group2_comparison, exact = FALSE)$p.value
} else {
    NA
}

# P-value 2: Compare GPT-4.1 vs Gemma on one-shot text
group3_comparison <- raw_data_for_stats$F1score[
    raw_data_for_stats$llm_clean == "GPT-4.1" & 
    raw_data_for_stats$prompt_clean == "one-shot" & 
    raw_data_for_stats$input_clean == "text"
]
group4_comparison <- raw_data_for_stats$F1score[
    raw_data_for_stats$llm_clean == "Gemma" & 
    raw_data_for_stats$prompt_clean == "one-shot" & 
    raw_data_for_stats$input_clean == "image"
]

group3_comparison <- group3_comparison[!is.na(group3_comparison)]
group4_comparison <- group4_comparison[!is.na(group4_comparison)]

pval2 <- if (length(group3_comparison) > 0 && length(group4_comparison) > 0) {
    wilcox.test(group3_comparison, group4_comparison, exact = FALSE)$p.value
} else {
    NA
}

# P-value 3: Compare GPT-4.1 vs Gemma on zero-shot image
group5_comparison <- raw_data_for_stats$F1score[
    raw_data_for_stats$llm_clean == "GPT-4.1" & 
    raw_data_for_stats$prompt_clean == "one-shot" & 
    raw_data_for_stats$input_clean == "text"
]
group6_comparison <- raw_data_for_stats$F1score[
    raw_data_for_stats$llm_clean == "Gemma" & 
    raw_data_for_stats$prompt_clean == "zero-shot" & 
    raw_data_for_stats$input_clean == "image"
]

group5_comparison <- group5_comparison[!is.na(group5_comparison)]
group6_comparison <- group6_comparison[!is.na(group6_comparison)]

pval3 <- if (length(group5_comparison) > 0 && length(group6_comparison) > 0) {
    wilcox.test(group5_comparison, group6_comparison, exact = FALSE)$p.value
} else {
    NA
}

# P-value 4: Compare GPT-4.1 on zero-shot text vs Gemma on one-shot image
group7_comparison <- raw_data_for_stats$F1score[
    raw_data_for_stats$llm_clean == "GPT-4.1" & 
    raw_data_for_stats$prompt_clean == "zero-shot" & 
    raw_data_for_stats$input_clean == "text"
]
group8_comparison <- raw_data_for_stats$F1score[
    raw_data_for_stats$llm_clean == "Gemma" & 
    raw_data_for_stats$prompt_clean == "one-shot" & 
    raw_data_for_stats$input_clean == "image"
]

group7_comparison <- group7_comparison[!is.na(group7_comparison)]
group8_comparison <- group8_comparison[!is.na(group8_comparison)]

pval4 <- if (length(group7_comparison) > 0 && length(group8_comparison) > 0) {
    wilcox.test(group7_comparison, group8_comparison, exact = FALSE)$p.value
} else {
    NA
}

# P-value 5: Compare Llama vs Mistral both zero-shot text
group9_comparison <- raw_data_for_stats$F1score[
    raw_data_for_stats$llm_clean == "Llama3.1" &
    raw_data_for_stats$prompt_clean == "zero-shot" &
    raw_data_for_stats$input_clean == "text"
]

group10_comparison <- raw_data_for_stats$F1score[
    raw_data_for_stats$llm_clean == "Mistral" &
    raw_data_for_stats$prompt_clean == "zero-shot" &
    raw_data_for_stats$input_clean == "text"
]

pval5 <- if (length(group9_comparison) > 0 && length(group10_comparison) > 0) {
    wilcox.test(group9_comparison, group10_comparison, exact = FALSE)$p.value
} else {
    NA
}


# P-value 6: Compare GliNER vs BioGliNER on zero-shot text
gliner_zst <- raw_data_for_stats$F1score[
    raw_data_for_stats$llm_clean == "GliNER" &
    raw_data_for_stats$prompt_clean == "zero-shot" &
    raw_data_for_stats$input_clean == "text"
]
biogliner_zst <- raw_data_for_stats$F1score[
    raw_data_for_stats$llm_clean == "BioGliNER" &
    raw_data_for_stats$prompt_clean == "zero-shot" &
    raw_data_for_stats$input_clean == "text"
]

gliner_zst <- gliner_zst[!is.na(gliner_zst)]
biogliner_zst <- biogliner_zst[!is.na(biogliner_zst)]

pval6 <- if (length(gliner_zst) > 0 && length(biogliner_zst) > 0) {
    wilcox.test(gliner_zst, biogliner_zst, exact = FALSE)$p.value
} else {
    NA
}

# P-value 7: Compare NuExtract vs GPT-4.1 on zero-shot text
nuextract_zst <- raw_data_for_stats$F1score[
    raw_data_for_stats$llm_clean == "NuExtract" &
    raw_data_for_stats$prompt_clean == "zero-shot" &
    raw_data_for_stats$input_clean == "text"
]
gpt_zst <- raw_data_for_stats$F1score[
    raw_data_for_stats$llm_clean == "GPT-4.1" &
    raw_data_for_stats$prompt_clean == "zero-shot" &
    raw_data_for_stats$input_clean == "text"
]

nuextract_zst <- nuextract_zst[!is.na(nuextract_zst)]
gpt_zst <- gpt_zst[!is.na(gpt_zst)]

pval7 <- if (length(nuextract_zst) > 0 && length(gpt_zst) > 0) {
    wilcox.test(nuextract_zst, gpt_zst, exact = FALSE)$p.value
} else {
    NA
}

# P-value 8: Compare GPT-4.1 vs Llama3.1 on zero-shot text
gpt_zst_llama <- raw_data_for_stats$F1score[
    raw_data_for_stats$llm_clean == "GPT-4.1" &
    raw_data_for_stats$prompt_clean == "zero-shot" &
    raw_data_for_stats$input_clean == "text"
]
llama_zst_gpt <- raw_data_for_stats$F1score[
    raw_data_for_stats$llm_clean == "Llama3.1" &
    raw_data_for_stats$prompt_clean == "zero-shot" &
    raw_data_for_stats$input_clean == "text"
]

gpt_zst_llama <- gpt_zst_llama[!is.na(gpt_zst_llama)]
llama_zst_gpt <- llama_zst_gpt[!is.na(llama_zst_gpt)]

pval8 <- if (length(gpt_zst_llama) > 0 && length(llama_zst_gpt) > 0) {
    wilcox.test(gpt_zst_llama, llama_zst_gpt, exact = FALSE)$p.value
} else {
    NA
}

# P-value 9: Compare NuExtract vs Llama3.1 on zero-shot text
nuextract_zst_llama <- raw_data_for_stats$F1score[
    raw_data_for_stats$llm_clean == "NuExtract" &
    raw_data_for_stats$prompt_clean == "zero-shot" &
    raw_data_for_stats$input_clean == "text"
]
llama_zst_nuext <- raw_data_for_stats$F1score[
    raw_data_for_stats$llm_clean == "Llama3.1" &
    raw_data_for_stats$prompt_clean == "zero-shot" &
    raw_data_for_stats$input_clean == "text"
]

nuextract_zst_llama <- nuextract_zst_llama[!is.na(nuextract_zst_llama)]
llama_zst_nuext <- llama_zst_nuext[!is.na(llama_zst_nuext)]

pval9 <- if (length(nuextract_zst_llama) > 0 && length(llama_zst_nuext) > 0) {
    wilcox.test(nuextract_zst_llama, llama_zst_nuext, exact = FALSE)$p.value
} else {
    NA
}

# Create vector of real p-values
pvals <- c(pval1, pval2, pval3, pval4, pval5, pval6, pval7, pval8, pval9)

# Remove NA p-values
pvals <- pvals[!is.na(pvals)]

# Print the calculated p-values for verification
cat("\nCalculated p-values:\n")
cat("P-value 1 - GPT-4.1 zero-shot text vs Gemma zero-shot image: ", pval1, "\n")
cat("P-value 2 - GPT-4.1 one-shot text vs Gemma one-shot image: ", pval2, "\n")
cat("P-value 3 - GPT-4.1 one-shot text vs Gemma zero-shot image: ", pval3, "\n")
cat("P-value 4 - GPT-4.1 zero-shot text vs Gemma one-shot image: ", pval4, "\n")
cat("P-value 5 - Llama vs Mistral Kruskal-Wallis: ", pval5, "\n")
cat("P-value 6 - GliNER vs BioGliNER zero-shot text: ", pval6, "\n")
cat("P-value 7 - NuExtract vs GPT-4.1 zero-shot text: ", pval7, "\n")
cat("P-value 8 - GPT-4.1 vs Llama3.1 zero-shot text: ", pval8, "\n")
cat("P-value 9 - NuExtract vs Llama3.1 zero-shot text: ", pval9, "\n")

# Define colors
mycolors <- c("firebrick2", "firebrick4", "steelblue2", "steelblue4")

# Use tryCatch to ensure dev.off() is called even if errors occur
tryCatch({
    png("F1_scores_plot.png", width = 10, height = 5, units = "in", res = 600)
    # Set margins and axis label orientations
    op <- par(mar = c(2, 4, 4, 2), cex = 1.2)

    # Draw plot
    xs <- barplot(
        t(f1_mat), beside = TRUE, space = c(0, 1.5),
        ylim = c(0, 100),
        col = mycolors, border = NA, 
        ylab = expression(F[1] ~ "score (%)"), 
        main = "F1 Score by Model and Condition",
        xaxt = "n"
    )

    # Add custom rotated x-axis labels
    x_coords <- colMeans(xs) + 0.5
    # Adjust y position to be below the plot area
    y_pos <- par("usr")[3] - 2 
    text(x = x_coords, y = y_pos, labels = rownames(f1_mat), srt = 0, adj = 1, xpd = TRUE)


    # Draw the error bars using real standard errors
    yogitools::errorBars(xs, t(f1_mat), t(stderr), l = .05)

    # Determine x-coordinates of missing values
    xna <- xs[apply(t(f1_mat), 1:2, is.na)]

    # Add hatching at missing value positions
    if (length(xna) > 0) {
        rect(xna - 0.5, 0, xna + 0.5, 100, col = "gray", density = 20, border = NA)
    }

    # Add significance brackets comparing GPT-4.1 vs Gemma across conditions
    # Find GPT-4.1 and Gemma positions in the matrix
    gpt_row <- which(mnames == "GPT-4.1")
    gemma_row <- which(mnames == "Gemma")

    print("Model positions:")
    print(paste("GPT-4.1 at row:", gpt_row))
    print(paste("Gemma at row:", gemma_row))

    # Column positions: zero-shot;text=1, one-shot;text=2, zero-shot;image=3, one-shot;image=4 # nolint: line_length_linter.
    if (length(pvals) >= 1 && !is.na(pvals[1]) && length(gpt_row) > 0 && length(gemma_row) > 0) { # nolint: line_length_linter.
        # Compare GPT vs Gemma on zero-shot text
        x1_gpt <- xs[1, gpt_row]
        x1_gemma <- xs[3, gemma_row]
        stars <- ifelse(pvals[1] < 0.001, "***", ifelse(pvals[1] < 0.01, "**", ifelse(pvals[1] < 0.05, "*", "ns")))
        
        if (stars != "ns") {
            y_pos <- 85
            segments(x1_gpt, y_pos, x1_gemma, y_pos)
            text((x1_gpt + x1_gemma) / 2, y_pos + 2, stars, cex = 1.5)
        }
    }

    if (length(pvals) >= 2 && !is.na(pvals[2]) && length(gpt_row) > 0 && length(gemma_row) > 0) { # nolint: line_length_linter.
        # Compare GPT vs Gemma on one-shot text
        x2_gpt <- xs[2, gpt_row]
        x2_gemma <- xs[4, gemma_row]
        stars <- ifelse(pvals[2] < 0.001, "***", ifelse(pvals[2] < 0.01, "**", ifelse(pvals[2] < 0.05, "*", "ns")))
        
        if (stars != "ns") {
            y_pos <- 80
            segments(x2_gpt, y_pos, x2_gemma, y_pos)
            text((x2_gpt + x2_gemma) / 2, y_pos + 2, stars, cex = 1.5)
        }
    }

    if (length(pvals) >= 3 && !is.na(pvals[3]) && length(gpt_row) > 0 && length(gemma_row) > 0) { # nolint: line_length_linter.
        # Compare GPT vs Gemma on zero-shot image
        x3_gpt <- xs[2, gpt_row]
        x3_gemma <- xs[3, gemma_row]
        stars <- ifelse(pvals[3] < 0.001, "***", ifelse(pvals[3] < 0.01, "**", ifelse(pvals[3] < 0.05, "*", "ns")))
        
        if (stars != "ns") {
            y_pos <- 75
            segments(x3_gpt, y_pos, x3_gemma, y_pos)
            text((x3_gpt + x3_gemma) / 2, y_pos + 2, stars, cex = 1.5)
        }
    }

    if (length(pvals) >= 4 && !is.na(pvals[4]) && length(gpt_row) > 0 && length(gemma_row) > 0) { # nolint: line_length_linter.
        # Compare GPT vs Gemma on one-shot image
        x4_gpt <- xs[1, gpt_row]
        x4_gemma <- xs[4, gemma_row]
        stars <- ifelse(pvals[4] < 0.001, "***", ifelse(pvals[4] < 0.01, "**", ifelse(pvals[4] < 0.05, "*", "ns")))
        
        if (stars != "ns") {
            y_pos <- 70
            segments(x4_gpt, y_pos, x4_gemma, y_pos)
            text((x4_gpt + x4_gemma) / 2, y_pos + 2, stars, cex = 1.5)
        }
    }

    #Pval 5 llama and mistral zero-shot text

    if (length(pvals) >= 5 && !is.na(pvals[5])) {
        llama_row <- which(mnames == "Llama3.1")
        mistral_row <- which(mnames == "Mistral")
        if (length(llama_row) > 0 && length(mistral_row) > 0) {
            x_llama_zst <- xs[1, llama_row]
            x_mistral_zst <- xs[1, mistral_row]
            
            stars <- ifelse(pvals[5] < 0.001, "***", ifelse(pvals[5] < 0.01, "**", ifelse(pvals[5] < 0.05, "*", "ns")))
            
            if (stars != "n") {
                y_pos <- 50
                segments(x_llama_zst, y_pos, x_mistral_zst, y_pos)
                text((x_llama_zst + x_mistral_zst) / 2, y_pos + 2, stars, cex = 1.5)
            }
        }
    }

    # P-value 6: GliNER vs BioGliNER on zero-shot text
    if (length(pvals) >= 6 && !is.na(pvals[6])) {
        gliner_row <- which(mnames == "GliNER")
        biogliner_row <- which(mnames == "BioGliNER")
        if (length(gliner_row) > 0 && length(biogliner_row) > 0) {
            x_gliner_zst <- xs[1, gliner_row]
            x_biogliner_zst <- xs[1, biogliner_row]
            
            stars <- ifelse(pvals[6] < 0.001, "***", ifelse(pvals[6] < 0.01, "**", ifelse(pvals[6] < 0.05, "*", "ns")))
            
            if (stars != "ns") {
                y_pos <- 65
                segments(x_gliner_zst, y_pos-5, x_biogliner_zst, y_pos-5)
                text((x_gliner_zst + x_biogliner_zst) / 2, y_pos - 3, stars, cex = 1.5)
            }
        }
    }

    # P-value 7: NuExtract vs GPT-4.1 on zero-shot text
    if (length(pvals) >= 7 && !is.na(pvals[7])) {
        nuextract_row <- which(mnames == "NuExtract")
        gpt_row <- which(mnames == "GPT-4.1")
        if (length(nuextract_row) > 0 && length(gpt_row) > 0) {
            x_nuextract_zst <- xs[1, nuextract_row]
            x_gpt_zst <- xs[1, gpt_row]
            
            stars <- ifelse(pvals[7] < 0.001, "***", ifelse(pvals[7] < 0.01, "**", ifelse(pvals[7] < 0.05, "*", "ns")))
            
            if (stars != "ns") {
                y_pos <- 55
                segments(x_nuextract_zst, y_pos, x_gpt_zst, y_pos)
                text((x_nuextract_zst + x_gpt_zst) / 2, y_pos + 2, stars, cex = 1.5)
            }
        }
    }

    # P-value 8: GPT-4.1 vs Llama3.1 on zero-shot text
    if (length(pvals) >= 8 && !is.na(pvals[8])) {
        gpt_row <- which(mnames == "GPT-4.1")
        llama_row <- which(mnames == "Llama3.1")
        if (length(gpt_row) > 0 && length(llama_row) > 0) {
            x_gpt_zst <- xs[1, gpt_row]
            x_llama_zst <- xs[1, llama_row]
            
            stars <- ifelse(pvals[8] < 0.001, "***", ifelse(pvals[8] < 0.01, "**", ifelse(pvals[8] < 0.05, "*", "ns")))
            
            if (stars != "ns") {
                y_pos <- 65
                segments(x_gpt_zst, y_pos, x_llama_zst, y_pos)
                text((x_gpt_zst + x_llama_zst) / 2, y_pos + 2, stars, cex = 1.5)
            }
        }
    }

    # P-value 9: NuExtract vs Llama3.1 on zero-shot text
    if (length(pvals) >= 9 && !is.na(pvals[9])) {
        nuextract_row <- which(mnames == "NuExtract")
        llama_row <- which(mnames == "Llama3.1")
        if (length(nuextract_row) > 0 && length(llama_row) > 0) {
            x_nuextract_zst <- xs[1, nuextract_row]
            x_llama_zst <- xs[1, llama_row]
            
            stars <- ifelse(pvals[9] < 0.001, "***", ifelse(pvals[9] < 0.01, "**", ifelse(pvals[9] < 0.05, "*", "ns")))
            
            if (stars != "ns") {
                y_pos <- 90
                segments(x_nuextract_zst, y_pos, x_llama_zst, y_pos)
                text((x_nuextract_zst + x_llama_zst) / 2, y_pos + 2, stars, cex = 1.5)
            }
        }
    }

    # Add grid lines
    grid(NA, NULL)

    # Add legend
    legend("topright", colnames(f1_mat), fill = mycolors, bg = "white")

    # Restore original parameters
    par(op)

}, finally = {
    # Ensure the PDF device is closed
    dev.off()
})

# Print matrices for verification
cat("\nF1 Score Matrix:\n")
print(f1_mat)
cat("\nStandard Error Matrix:\n")
print(stderr_mat)