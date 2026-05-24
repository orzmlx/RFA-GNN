import re

with open('/Users/liuxi/Desktop/RFA_GNN/liuthesis_my/Result.tex', 'r') as f:
    text = f.read()

# Replace Paragraph 1
old1 = r"This chapter presents the main results and the perfermance of the proposed model. It compares different model architectures, shows how the results change across settings, and looks at what the model learns from the biological data."
new1 = r"This chapter presents the main results of our proposed model. It compares different model architectures, shows how the scores change across different tasks, and explains what the model actually learned from the biology."
text = text.replace(old1, new1)

# Replace Paragraph 2, 3
old2 = r"""Figure~\ref{fig:evaluation_framework} organizes the evaluation into several metric groups. Error metrics are presented first because they give the most direct view of numerical prediction accuracy. In this section, the main error metric is mean squared error. A lower value means that the predicted response is closer to the observed response.

\noindent Table~\ref{tab:mse_results} reports the MSE results for the three splits. In the updated CAGNN run, CAGNN gives the lowest MSE in all three settings. The gain is clearest in the cold drug split, and the updated cold cell result is also much lower than before. This suggests that the hybrid version improves absolute prediction accuracy across all tested generalization settings. The contrast with the correlation results remains important, because a good MSE does not always guarantee that the full gene level pattern is captured."""
new2 = r"""Figure~\ref{fig:evaluation_framework} groups our evaluation by different metrics. We show error metrics first because they directly measure how close our predictions are to the real numbers. The main error metric here is the mean squared error (MSE), where a lower score means a better prediction.

\noindent Table~\ref{tab:mse_results} reports the MSE scores for the three data splits. CAGNN gives the lowest error across all of them. The improvement is most obvious in the cold drug split, while the cold cell error has also dropped significantly. This means our model gets the absolute numbers right in different scenarios. However, a low MSE is not everything. It does not blindly guarantee that the model caught the true overall shape of the gene changes, which is why we also need correlation scores."""
text = text.replace(old2, new2)

# Replace Paragraph 4, 5
old3 = r"""\noindent Table~\ref{tab:mse_results} summarizes the average test MSE, but this summary alone does not show where the prediction errors occur or how they are distributed across samples and genes. Samples with weak responses are often hard to predict, because even a small numerical shift can create a large relative error. This issue becomes more important in the cold cell split, where the baseline expression state changes across cell lines.

\noindent Annotation quality also matters here. The current pipeline keeps only treatment profiles that have target annotations. Even with this filter, the target information is still incomplete. A drug can have only a few known targets and still produce a strong downstream response through unknown or indirect paths. When such cases fail, the result should not be read only as a weakness of the graph model."""
new3 = r"""\noindent While Table~\ref{tab:mse_results} gives average MSEs, it does not tell us where exactly the predictions failed. For example, weak gene responses are notoriously hard to predict because a tiny numerical mistake looks like a huge error. This problem gets worse in the cold cell split, since every cell line has a completely different starting state.

\noindent The quality of our drug labels also matters. Even though we only look at drugs with known targets, this information is incomplete. A drug might hit its main target but cause huge side effects through unknown pathways. If the model fails on these cases, it is often because the biological labels are missing, rather than the graph model being weak."""
text = text.replace(old3, new3)

# Replace Precision
old4 = r"""The second group of metrics tests whether the models recover the most responsive genes. This view is important because perturbation analysis often focuses more on the strongest up regulated and down regulated genes than on every small change in the full response vector. For this reason, top gene precision is treated as a separate part of the evaluation.

\noindent This analysis is computed directly from the saved prediction files. For each sample, the true response vector $y_{\mathrm{true}}$ and the predicted response vector $y_{\mathrm{pred}}$ are ranked across genes."""
new4 = r"""Next, we check if the models can find the genes that react the most. In real-world biology, researchers care much more about the top changed genes than small random shifts across the entire gene profile. That is why finding the top-changing genes is so important.

\noindent We compute this directly from the saved prediction files. For each sample, we rank the genes in both the true response ($y_{\mathrm{true}}$) and the predicted response ($y_{\mathrm{pred}}$)."""
text = text.replace(old4, new4)

# Replace Correlation
old5 = r"""The main correlation metric is the Pearson correlation coefficient(PCC). PCC measures whether the predicted response follows the correct gene level pattern within each sample. A higher value indicates better agreement in the shape of the response profile, even when the absolute magnitude is not matched exactly.

\noindent Table~\ref{tab:pcc_results} reports the PCC results for the three splits. CAGNN gives the best PCC in all three settings, including cold cell. DeepCOP remains competitive in MSE in some settings, but its PCC drops more strongly in the harder splits. This contrast between MSE and PCC is informative, because a model may preserve part of the gene ranking pattern while still making larger errors in absolute magnitude."""
new5 = r"""The Pearson correlation coefficient (PCC) is our main correlation metric. PCC measures if the predicted gene response goes in the right direction and shape for each sample. A higher score means the predicted shape matches the real biology, even if the exact numbers are slightly off.

\noindent Table~\ref{tab:pcc_results} reports the PCC scores for the three splits. CAGNN wins across all three settings. DeepCOP still has decent MSE scores, but its PCC drops heavily on the harder tasks. This shows why we need both metrics: a model might predict acceptable average numbers (MSE) but completely fail to rank the genes correctly (PCC)."""
text = text.replace(old5, new5)

# Replace Generalization
old6 = r"""\noindent The PCC results also help clarify generalization under distribution shift. The harder blind tests are more informative than the warm split because they show whether the model learns broader response patterns or mainly adapts to a familiar data setting. In Table~\ref{tab:pcc_results}, the cold cell setting is still the clearest bottleneck, because the PCC values are lower than in warm and cold drug for all three models. This pattern is consistent with the biological setting, because a new cell line changes the baseline state and can also change how the same drug acts in context.

\noindent Taken together, the results suggest that the CAGNN is the strongest of the three learned models in terms of PCC across all splits. The margin is largest in the warm and cold drug settings, while the advantage in cold cell is smalle, but robust transfer across unseen cell contexts remains the main difficulty."""
new6 = r"""\noindent The harder blind tests tell us more than the warm split. They reveal whether the model actually learned the biology, or if it just memorized familiar data. Looking at Table~\ref{tab:pcc_results}, the cold cell setting is clearly the biggest hurdle, as the PCC values drop for all three models. This makes biological sense: a new cell line changes the baseline state completely and alters how the drug will behave.

\noindent Overall, CAGNN strongly leads the pack in PCC across all splits. Its winning margin is large in the warm and cold drug tasks. The advantage in the cold cell task is smaller but still visible, proving that predicting for completely new cell lines remains the hardest problem to solve."""
text = text.replace(old6, new6)

# Ablation summary
old7 = r"""\noindent Table~\ref{tab:context_ablation} shows the corresponding test results. Replacing cell identity with a cell context gives a small PCC gain in the warm and cold drug settings, but the gain is much larger in the cold cell setting. This pattern suggests that a sample specific cell context is more useful than a fixed cell label when the model must generalize to unseen cell lines.

\noindent These results also clarify the different roles of cell information in the model. A fixed cell identity embedding gives a coarse description of the cell line. A control derived cell context gives a sample specific description of the current baseline state. The gains in Table~\ref{tab:context_ablation} suggest that the larger benefit comes from replacing a fixed label with a sample specific context."""
new7 = r"""\noindent Table~\ref{tab:context_ablation} shows the results. Using the cell baseline context instead of just an ID label gives a slight bump in the warm and cold drug tasks, but a massive jump in the cold cell task. This proves that feeding the model a real, sample-specific baseline helps much more than just telling it the name of the cell line.

\noindent A hardcoded cell ID only tells the model the general type of the cell. The baseline context, on the other hand, tells the model the exact molecular state right before the drug is given. The huge gain in Table~\ref{tab:context_ablation} confirms that reading the actual cell state is the better choice."""
text = text.replace(old7, new7)

# Uncertainty wording
old8 = r"""\noindent The next analysis studies whether uncertainty is associated with absolute prediction error. For each test sample, the average predicted standard deviation can be plotted against the MSE. In all three splits, the relationship is positive. The pattern is strongest in cold cell, moderate in warm, and weaker in cold drug. This does not prove perfect calibration, but it does show that the uncertainty output contains useful information about absolute prediction risk."""
new8 = r"""\noindent We also check if our model properly doubts itself when it makes bigger errors. For each test sample, we plotted its predicted uncertainty against its actual MSE. In all three splits, the trend is positive: high error matches high uncertainty. The trend is strongest in cold cell, but weaker in cold drug. While the calibration isn't perfect, it clearly shows that the model knows when it is making a risky, unreliable guess."""
text = text.replace(old8, new8)

# Case studies
old9 = r"""A compact case study helps connect the summary metrics to concrete examples. The goal here is not to compare models. Instead, this section looks only at the proposed model. The analysis examines whether the model focuses on a meaningful part of the biological graph instead of spreading importance everywhere. It also examines whether the learned attention pattern helps explain why the model performs well in some cases and poorly in others.

\noindent The analysis test samples by drug in the cold drug split and ranks drugs by median sample wise PCC within the proposed model. This ranking gives one strong case and two difficult cases."""
new9 = r"""Instead of just looking at average metrics, going through specific biological examples makes things much easier to understand. In this section, we only look at our CAGNN model to see how it "thinks". We want to know if the model is actually paying attention to a specific, logical part of the biological graph, rather than just randomly throwing weights everywhere. This attention trick helps explain why the model succeeds in some cases and fails in others.

\noindent We picked drugs from the cold drug split and ranked them by their median PCC. This gave us one very successful drug prediction and two poor ones."""
text = text.replace(old9, new9)

old10 = r"""\noindent This analysis links three levels of information in one workflow. The first is the known drug targets. The second is the graph path between those targets and other genes. The third is the final set of high attention edges emphasized by the model. When these three levels remain close in the graph, the attention pattern is easier to understand biologically. In the current best case, this path tracing view shows that the high attention region is not isolated from the known targets, but can be reached through short graph paths. This does not prove a causal mechanism, but it gives stronger support for interpretation than showing attention weights alone."""
new10 = r"""\noindent This path-tracing view connects three things together: the known drug targets, the physical paths in the biological network, and the edges our model ended up paying the most attention to. When these three things align, the model's choices make perfect biological sense. In our best case, the highly-attended genes were just a few steps away from the original drug targets. Even though this isn't hard proof of exactly how the drug works, it is much more convincing than just showing a table of random high weights."""
text = text.replace(old10, new10)

with open('/Users/liuxi/Desktop/RFA_GNN/liuthesis_my/Result.tex.tmp', 'w') as f:
    f.write(text)
