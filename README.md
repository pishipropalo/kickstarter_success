# kickstarter_success

[RESULTS.md](https://github.com/user-attachments/files/20605906/RESULTS.md)
# В конце скрипта добавьте:
with open('README.md', 'w') as f:
    f.write("# Kickstarter Success Prediction\n\n")
    f.write("## Latest Model Comparison Results\n\n")
    f.write(metrics_df.to_markdown(index=False))
    f.write("\n\n[Full details](RESULTS.md)")
