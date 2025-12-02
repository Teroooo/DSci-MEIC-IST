# ------------------
# Correlation analysis
# ------------------

if credit_score_vars and run_credit_score_correlation_analysis:
    print("Printing correlation analysis for credit score...")

    encoded_credit_score_filename = "../../classification/traffic_accidents_encoded.csv"
    encoded_credit_score_data: DataFrame = read_csv(encoded_credit_score_filename, na_values="")

    credit_score_corr_mtx: DataFrame = encoded_credit_score_data.corr().abs()

    plt.figure(figsize=(10, 9))
    heatmap(
        abs(credit_score_corr_mtx),
        xticklabels=credit_score_corr_mtx.columns,
        yticklabels=credit_score_corr_mtx.columns,
        annot=False,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    plt.tight_layout()
    print("Saving image for credit score correlation analysis...")
    plt.savefig(f"{credit_score_savefig_path_prefix}_correlation_analysis.png")
    plt.show()
    print("Image saved")
    plt.clf()
else:
    if not credit_score_vars:
        print("Correlation analysis: there are no variables.")
    if not run_credit_score_correlation_analysis:
        print("Correlation analysis: skipping.")