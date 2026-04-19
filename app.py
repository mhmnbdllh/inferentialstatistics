
    # ── TABS ─────────────────────────────────────────────────
    t1,t2,t3,t4 = st.tabs([
        "📂 Data Input","🔬 Assumption Tests","📊 Test Results","📚 Statistical Guide"])

    # ── TAB 1: DATA INPUT ────────────────────────────────────
    with t1:
        st.markdown('<div class="sec-head">Upload CSV File</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        col_up, col_cfg = st.columns([3,2])

        with col_up:
            st.markdown('<div class="sec-head">Demo Datasets</div>', unsafe_allow_html=True)
            demo = st.selectbox("Select demo",
                ["One-Sample (normal)","One-Sample (non-normal)",
                 "Paired (normal)","Paired (non-normal)",
                 "Independent (normal, equal var)","Independent (normal, unequal var)",
                 "Independent (non-normal)"])
            if st.button("▶  Load Demo & Run"):
                np.random.seed(42)
                dmu0 = 0.0
                if demo=="One-Sample (normal)":
                    ddf=pd.DataFrame({"value":np.random.normal(52,8,30)}); dmu0=50.0
                elif demo=="One-Sample (non-normal)":
                    ddf=pd.DataFrame({"value":np.random.exponential(5,25)}); dmu0=4.0
                elif demo=="Paired (normal)":
                    p=np.random.normal(70,10,25)
                    ddf=pd.DataFrame({"pre":p,"post":p-np.random.normal(5,3,25)})
                elif demo=="Paired (non-normal)":
                    p=np.random.exponential(10,20)
                    ddf=pd.DataFrame({"pre":p,"post":p*np.random.uniform(0.7,0.95,20)})
                elif demo=="Independent (normal, equal var)":
                    ddf=pd.DataFrame({"value":np.concatenate([np.random.normal(65,8,25),np.random.normal(72,8,25)]),"group":["A"]*25+["B"]*25})
                elif demo=="Independent (normal, unequal var)":
                    ddf=pd.DataFrame({"value":np.concatenate([np.random.normal(65,5,20),np.random.normal(72,15,20)]),"group":["A"]*20+["B"]*20})
                else:
                    ddf=pd.DataFrame({"value":np.concatenate([np.random.exponential(5,20),np.random.exponential(8,20)]),"group":["A"]*20+["B"]*20})
                ttype_map={"One-Sample (normal)":"One-Sample","One-Sample (non-normal)":"One-Sample",
                           "Paired (normal)":"Paired (Dependent) Samples","Paired (non-normal)":"Paired (Dependent) Samples",
                           "Independent (normal, equal var)":"Independent Samples",
                           "Independent (normal, unequal var)":"Independent Samples","Independent (non-normal)":"Independent Samples"}
                st.session_state.update({"df_input":ddf,"run_auto":True,
                                         "test_type_sel":ttype_map[demo],
                                         "alpha_sel":alpha,"alt_sel":alternative,
                                         "demo_mu0":dmu0,"cfg":{}})
                st.rerun()

        if uploaded:
            df_input=pd.read_csv(uploaded)
            st.session_state["df_input"]=df_input

        if "df_input" in st.session_state:
            df_input=st.session_state["df_input"]
            with col_up:
                st.markdown('<div class="sec-head">Data Preview</div>', unsafe_allow_html=True)
                st.dataframe(df_input.head(30),use_container_width=True)
                st.markdown(f'<div class="info-box">📌 <b>{len(df_input)}</b> rows × <b>{len(df_input.columns)}</b> columns loaded.</div>',
                            unsafe_allow_html=True)

            with col_cfg:
                st.markdown('<div class="sec-head">Column Mapping</div>', unsafe_allow_html=True)
                num_cols=df_input.select_dtypes(include=[np.number]).columns.tolist()
                all_cols=df_input.columns.tolist()
                ttype=st.session_state.get("test_type_sel",test_type)
                alph =st.session_state.get("alpha_sel",alpha)
                alt  =st.session_state.get("alt_sel",alternative)
                cfg={}
                if "One-Sample" in ttype:
                    cfg["value_col"]=st.selectbox("Value Column",num_cols,key="vc")
                    cfg["mu0"]=st.number_input("Hypothesised Mean (μ₀)",
                                               value=st.session_state.get("demo_mu0",0.0),
                                               step=0.01,format="%.4f")
                elif "Paired" in ttype:
                    cfg["pre_col"] =st.selectbox("Pre / Group 1 Column",num_cols,key="pc1")
                    rem=[c for c in num_cols if c!=cfg.get("pre_col")]
                    cfg["post_col"]=st.selectbox("Post / Group 2 Column",
                                                  rem if rem else num_cols,key="pc2")
                else:
                    cfg["value_col"]=st.selectbox("Value Column",num_cols,key="vc2")
                    cfg["group_col"]=st.selectbox("Group Column",all_cols,key="gc")
                if st.button("▶  Run Analysis",key="run_btn"):
                    st.session_state.update({"cfg":cfg,"test_type_sel":ttype,
                                             "alpha_sel":alph,"alt_sel":alt,
                                             "run_auto":True})
                    st.rerun()
        else:
            st.markdown('<div class="info-box">⬆ Upload a CSV file or load a demo dataset above.</div>',
                        unsafe_allow_html=True)

    # ── ANALYSIS ENGINE ──────────────────────────────────────
    if st.session_state.get("run_auto") and "df_input" in st.session_state:
        df   = st.session_state["df_input"]
        cfg  = st.session_state.get("cfg",{})
        ttype= st.session_state.get("test_type_sel","One-Sample")
        alph = st.session_state.get("alpha_sel",0.05)
        alt  = st.session_state.get("alt_sel","two-sided")

        # Auto-build cfg for demos
        if not cfg:
            nc=df.select_dtypes(include=[np.number]).columns
            if "One-Sample" in ttype:
                cfg={"value_col":nc[0],"mu0":st.session_state.get("demo_mu0",0.0)}
            elif "Paired" in ttype:
                cfg={"pre_col":nc[0],"post_col":nc[1] if len(nc)>1 else nc[0]}
            else:
                cfg={"value_col":nc[0],"group_col":df.columns[-1]}
            st.session_state["cfg"]=cfg

        try:
            assump_dict={}; norm_figs=[]; norm_bytes_list=[]
            test_res=None; test_fig=None; desc_out={}; col_names=[]

            if "One-Sample" in ttype:
                data = df[cfg["value_col"]].dropna().values
                mu0  = float(cfg.get("mu0",0.0))
                col_names=[cfg["value_col"],f"μ₀={mu0}"]
                desc_out=descriptives(data)
                norm=normality_test(data,alph); assump_dict["Normality Test"]=norm
                nf=plot_normality(data,cfg["value_col"],norm)
                norm_figs.append(nf); norm_bytes_list.append(fig_bytes(nf))
                test_res=(one_sample_t(data,mu0,alph,alt)
                          if norm["normal"] else wilcoxon_one_sample(data,mu0,alph,alt))
                test_fig=plot_one_sample(data,test_res,mu0,alph)

            elif "Paired" in ttype:
                g1=df[cfg["pre_col"]].dropna().values
                g2=df[cfg["post_col"]].dropna().values
                n_min=min(len(g1),len(g2)); g1,g2=g1[:n_min],g2[:n_min]
                lbl1,lbl2=cfg["pre_col"],cfg["post_col"]
                col_names=[lbl1,lbl2]
                d=g1-g2
                desc_out={lbl1:descriptives(g1),lbl2:descriptives(g2),"Differences":descriptives(d)}
                norm=normality_test(d,alph); assump_dict["Normality of Differences"]=norm
                nf=plot_normality(d,f"{lbl1} − {lbl2}",norm)
                norm_figs.append(nf); norm_bytes_list.append(fig_bytes(nf))
                test_res=(paired_t(g1,g2,alph,alt)
                          if norm["normal"] else wilcoxon_paired(g1,g2,alph,alt))
                test_fig=plot_paired(g1,g2,test_res,lbl1,lbl2,alph)

            else:  # Independent Samples
                vc=cfg["value_col"]; gc=cfg["group_col"]
                grps=df[gc].dropna().unique()
                if len(grps)<2: st.error("Need at least 2 groups."); st.stop()
                if len(grps)>2: st.warning(f"Using first 2 groups: {grps[0]}, {grps[1]}")
                g1=df[df[gc]==grps[0]][vc].dropna().values
                g2=df[df[gc]==grps[1]][vc].dropna().values
                lbl1,lbl2=str(grps[0]),str(grps[1]); col_names=[lbl1,lbl2]
                desc_out={lbl1:descriptives(g1),lbl2:descriptives(g2)}
                nm1=normality_test(g1,alph); nm2=normality_test(g2,alph)
                assump_dict[f"Normality — {lbl1}"]=nm1
                assump_dict[f"Normality — {lbl2}"]=nm2
                nf1=plot_normality(g1,f"{vc} [{lbl1}]",nm1)
                nf2=plot_normality(g2,f"{vc} [{lbl2}]",nm2)
                norm_figs+=[nf1,nf2]; norm_bytes_list+=[fig_bytes(nf1),fig_bytes(nf2)]
                if nm1["normal"] and nm2["normal"]:
                    lev=levene_test(g1,g2,alph)
                    assump_dict["Levene's Test (Equality of Variances)"]=lev
                    test_res=independent_t(g1,g2,alph,alt)
                else:
                    test_res=mann_whitney_u(g1,g2,alph,alt)
                test_fig=plot_independent(g1,g2,test_res,lbl1,lbl2,alph)

            st.session_state.update({
                "assump_dict":assump_dict,"test_res":test_res,
                "test_fig":test_fig,"test_fig_bytes":fig_bytes(test_fig),
                "norm_figs":norm_figs,"norm_bytes":norm_bytes_list,
                "desc_out":desc_out,"col_names":col_names,"run_auto":False})

        except Exception as ex:
            import traceback
            st.error(f"Analysis error: {ex}")
            st.code(traceback.format_exc())
            st.session_state["run_auto"]=False

    # ── TAB 2: ASSUMPTIONS ───────────────────────────────────
    with t2:
        if "assump_dict" not in st.session_state:
            st.markdown('<div class="info-box">Run an analysis from the <b>Data Input</b> tab first.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="sec-head">Prerequisite Assumption Checks</div>',
                        unsafe_allow_html=True)
            for aname,ares in st.session_state["assump_dict"].items():
                is_ok=ares.get("normal",ares.get("equal_variance",True))
                badge=('<span class="badge-pass">✓ PASSED</span>' if is_ok
                       else '<span class="badge-fail-b">✗ FAILED</span>')
                with st.expander(f"{aname}  —  {badge}", expanded=True):
                    c1,c2,c3,c4=st.columns(4)
                    slbl=ares.get("stat_label","Stat")
                    with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Test</div><div class="metric-val" style="font-size:.95rem">{ares["test"]}</div></div>',unsafe_allow_html=True)
                    with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">{slbl} Statistic</div><div class="metric-val">{ares["statistic"]:.5f}</div></div>',unsafe_allow_html=True)
                    with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">p-value</div><div class="metric-val">{ares["p_value"]:.5f}</div></div>',unsafe_allow_html=True)
                    with c4:
                        clsc="green" if is_ok else "red"
                        st.markdown(f'<div class="metric-card {clsc}"><div class="metric-label">Result</div><div class="metric-val" style="font-size:.9rem">{ares["interpretation"]}</div></div>',unsafe_allow_html=True)
                    msg=("✅ <b>Parametric test applicable.</b> Normality assumption satisfied." if is_ok
                         else "⚠️ <b>Non-parametric equivalent applied.</b> Normality assumption violated.")
                    box="info-box" if is_ok else "warn-box"
                    st.markdown(f'<div class="{box}">{msg}</div>',unsafe_allow_html=True)

            st.markdown('<div class="sec-head">Normality Diagnostic Plots</div>', unsafe_allow_html=True)
            for nf in st.session_state["norm_figs"]:
                st.pyplot(nf,use_container_width=True); plt.close(nf)

    # ── TAB 3: RESULTS ───────────────────────────────────────
    with t3:
        if "test_res" not in st.session_state:
            st.markdown('<div class="info-box">Run an analysis from the <b>Data Input</b> tab first.</div>',
                        unsafe_allow_html=True)
        else:
            r    =st.session_state["test_res"]
            alph =st.session_state.get("alpha_sel",0.05)

            # Verdict
            if r["reject"]:
                st.markdown(f'<div class="verdict-banner verdict-reject"><div class="verdict-label">Decision at α = {alph}</div><div class="verdict-text-reject">✗ REJECT H₀</div><div class="verdict-sub">Sufficient statistical evidence to reject the null hypothesis.</div></div>',unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-banner verdict-fail"><div class="verdict-label">Decision at α = {alph}</div><div class="verdict-text-fail">✓ FAIL TO REJECT H₀</div><div class="verdict-sub">Insufficient statistical evidence to reject the null hypothesis.</div></div>',unsafe_allow_html=True)

            # Key metrics
            sl=("t" if "t-Test" in r["test"] else "W" if "Wilcoxon" in r["test"] else "U")
            pd_=f"{r['p_value']:.4f}" if r['p_value']>=0.0001 else "< 0.0001"
            ei=interpret_effect(r["effect_size"],r.get("effect_label",""))
            c1,c2,c3,c4=st.columns(4)
            with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Test Statistic ({sl})</div><div class="metric-val">{r["statistic"]:.4f}</div></div>',unsafe_allow_html=True)
            with c2:
                cc="red" if r["reject"] else "green"
                st.markdown(f'<div class="metric-card {cc}"><div class="metric-label">p-value</div><div class="metric-val">{pd_}</div></div>',unsafe_allow_html=True)
            with c3:
                if "df" in r: st.markdown(f'<div class="metric-card blue"><div class="metric-label">df</div><div class="metric-val">{r["df"]:.4f}</div></div>',unsafe_allow_html=True)
                else:
                    ne=r.get("n_effective",r.get("n","—"))
                    st.markdown(f'<div class="metric-card blue"><div class="metric-label">Effective N</div><div class="metric-val">{ne}</div></div>',unsafe_allow_html=True)
            with c4: st.markdown(f'<div class="metric-card"><div class="metric-label">{r.get("effect_label","Effect")}</div><div class="metric-val">{r["effect_size"]:.4f}</div><div class="metric-sub">{ei}</div></div>',unsafe_allow_html=True)

            if "ci" in r:
                st.markdown(f'<div class="info-box">📐 {int((1-alph)*100)}% Confidence Interval of the Difference: <b>[{r["ci"][0]:.4f}, {r["ci"][1]:.4f}]</b></div>',unsafe_allow_html=True)

            # SPSS-style tables
            st.markdown('<div class="sec-head">SPSS-Style Output Tables</div>', unsafe_allow_html=True)
            desc=st.session_state.get("desc_out",{})
            if desc:
                st.markdown("**Descriptive Statistics**")
                first=list(desc.values())[0]
                if isinstance(first,dict):
                    grps=list(desc.keys()); sk=list(list(desc.values())[0].keys())
                    rows=[["Statistic"]+grps]+[[s]+[str(desc[g][s]) for g in grps] for s in sk]
                else:
                    rows=[["Statistic","Value"]]+[[k,str(v)] for k,v in desc.items()]
                st.markdown(spss_table(rows),unsafe_allow_html=True)

            if "t_equal" in r:  # Independent t-test
                st.markdown("**Levene's Test for Equality of Variances**")
                lv=r["levene"]
                st.markdown(spss_table([["","F","df1","df2","Sig."],
                    ["Levene's Test",f"{lv['statistic']:.4f}",str(lv['df1']),
                     str(lv['df2']),f"{lv['p_value']:.4f}"]]),unsafe_allow_html=True)
                cp=int((1-alph)*100)
                st.markdown("**t-Test for Equality of Means**")
                st.markdown(spss_table([
                    ["","t","df","Sig.(2-tailed)",f"{cp}% CI Lower",f"{cp}% CI Upper","Used"],
                    ["Equal variances assumed",
                     f"{r['t_equal']:.4f}",f"{r['df_equal']:.2f}",f"{r['p_equal']:.4f}",
                     f"{r['ci_equal_lower']:.4f}",f"{r['ci_equal_upper']:.4f}",
                     "✓" if r["use_equal"] else ""],
                    ["Equal variances NOT assumed (Welch's)",
                     f"{r['t_welch']:.4f}",f"{r['df_welch']:.4f}",f"{r['p_welch']:.4f}",
                     f"{r['ci_welch_lower']:.4f}",f"{r['ci_welch_upper']:.4f}",
                     "" if r["use_equal"] else "✓"]]),unsafe_allow_html=True)
            elif "Mann-Whitney" in r["test"]:  # Mann-Whitney U (SPSS-style)
                st.markdown(f"**{r['test']}**")
                srows=[["Parameter","Value"]]
                
                # SPSS-style output
                srows.append(["Mann-Whitney U", f"{r['Mann_Whitney_U']:.4f}"])
                srows.append(["Wilcoxon W", f"{max(r['R1'], r['R2']):.4f}"])
                srows.append(["Test Statistic", f"{r['statistic']:.4f}"])
                srows.append(["Standard Error", f"{np.sqrt(r['n1']*r['n2']*(r['n1']+r['n2']+1)/12):.4f}"])
                srows.append(["Standardized Test Statistic (Z)", f"{r['Z']:.4f}"])
                srows.append(["Asymptotic Sig. (2-sided)", f"{r['p_value_asymptotic']:.5f}" if r['p_value_asymptotic'] >= 0.0001 else "< 0.0001"])
                if r.get('p_value_exact') is not None:
                    srows.append(["Exact Sig. (2-sided)", f"{r['p_value_exact']:.5f}" if r['p_value_exact'] >= 0.0001 else "< 0.0001"])
                srows.append(["Effect Size (r)", f"{r['effect_size']:.4f}"])
                srows.append(["Effect Magnitude", interpret_effect(r["effect_size"], r["effect_label"])])
                
                st.markdown(spss_table(srows), unsafe_allow_html=True)
            else:
                st.markdown(f"**{r['test']}**")
                srows=[["Parameter","Value"]]
                dmap={"statistic":"Test Statistic","df":"df","p_value":"Sig.",
                      "mean_diff":"Mean Difference","std_diff":"Std. Dev. of Differences",
                      "n":"N","n_effective":"N (effective)",
                      "T_plus":"T+ (Positive Ranks)","T_minus":"T− (Negative Ranks)",
                      "Z":"Z (asymptotic)","U1":"U₁","U2":"U₂",
                      "pearson_r":"Pearson r","pearson_p":"r Sig.",
                      "mean1":"Mean (Group 1)","mean2":"Mean (Group 2)",
                      "median1":"Median (Group 1)","median2":"Median (Group 2)"}
                for k,lbl in dmap.items():
                    if k in r and r[k] is not None and not isinstance(r[k],bool):
                        v=r[k]; srows.append([lbl,f"{v:.4f}" if isinstance(v,float) else str(v)])
                if "ci" in r:
                    srows.append([f"{int((1-alph)*100)}% CI Lower",f"{r['ci'][0]:.4f}"])
                    srows.append([f"{int((1-alph)*100)}% CI Upper",f"{r['ci'][1]:.4f}"])
                srows.append(["Effect Magnitude",ei])
                st.markdown(spss_table(srows),unsafe_allow_html=True)

            if "pearson_r" in r and "t-Test" in r["test"]:
                st.markdown("**Paired Samples Correlations**")
                st.markdown(spss_table([["Pair","N","Pearson r","Sig."],
                    ["Pair 1",str(r["n"]),f"{r['pearson_r']:.4f}",f"{r['pearson_p']:.4f}"]]),
                    unsafe_allow_html=True)

            # Plot
            st.markdown('<div class="sec-head">Visualisation</div>', unsafe_allow_html=True)
            st.pyplot(st.session_state["test_fig"],use_container_width=True)
            try: plt.close(st.session_state["test_fig"])
            except: pass

            # Downloads
            st.markdown('<div class="sec-head">Download Results</div>', unsafe_allow_html=True)
            dl1,dl2,dl3=st.columns(3)
            with dl1:
                st.download_button("⬇ Download Plot (PNG)",
                    data=st.session_state['test_fig_bytes'],
                    file_name="test_results.png",
                    mime="image/png"
                )
            
            # CSV download
            with dl2:
                csv_data = export_csv(
                    test_result=r,
                    assump_dict=st.session_state["assump_dict"],
                    desc_dict=st.session_state["desc_out"],
                    alpha=alph,
                    alternative=r.get("alternative", "two-sided")
                )
                st.download_button(
                    "⬇  Download Results (CSV)",
                    data=csv_data,
                    file_name="test_results.csv",
                    mime="text/csv"
                )
            
            # PDF download
            with dl3:
                pdf_bytes = build_pdf(
                    test_type=st.session_state.get("test_type_sel", "Independent Samples"),
                    assump_dict=st.session_state["assump_dict"],
                    test_result=r,
                    desc_dict=st.session_state["desc_out"],
                    norm_bytes_list=st.session_state["norm_bytes"],
                    fig_test_bytes=st.session_state["test_fig_bytes"],
                    alpha=alph,
                    alternative=r.get("alternative", "two-sided"),
                    col_names=st.session_state["col_names"]
                )
                st.download_button(
                    "⬇  Download Report (PDF)",
                    data=pdf_bytes,
                    file_name="statistical_report.pdf",
                    mime="application/pdf"
                )

    # ── TAB 4: USER GUIDE ────────────────────
    with t4:
        st.markdown("## User Guide — StatTest Suite")
        
        st.markdown("""
<div class="info-box">
This application automates the selection and execution of parametric and non-parametric hypothesis tests, 
including prerequisite assumption checking. All decisions are made algorithmically based on your data.
</div>
""", unsafe_allow_html=True)
        
        st.markdown("### 1. Conceptual Overview")
        st.markdown("""
Statistical hypothesis testing is a formal procedure for evaluating evidence against a null hypothesis (H₀). 
The **parametric** approach assumes the data follow a specific probability distribution (typically Gaussian/Normal). 
When this assumption is violated, **non-parametric** methods — which rely on rank-ordering rather than distributional 
assumptions — are employed as robust alternatives.
""")
        
        st.markdown("### 2. Decision Logic")
        st.markdown("""
<div class="formula-box">
STEP 1: Upload CSV → select test type and configuration
STEP 2: System runs normality test automatically
         • n < 50  → Shapiro-Wilk (W statistic)
         • n ≥ 50  → Kolmogorov-Smirnov (D statistic)
STEP 3: Decision branch
         • Normally distributed → Parametric test
         • Non-normal          → Non-parametric equivalent
STEP 4: For independent samples, if normal:
         • Run Levene's test for equality of variances
         • Equal variances     → Student's t-test (pooled)
         • Unequal variances   → Welch's t-test (Satterthwaite df)
</div>
""", unsafe_allow_html=True)
        
        st.markdown("### 3. Hypothesis Statements")
        
        with st.expander("One-Sample Tests", expanded=True):
            st.markdown("""
**Parametric: One-Sample t-Test**

H₀: μ = μ₀  
H₁: μ ≠ μ₀ (two-sided) | μ > μ₀ (greater) | μ < μ₀ (less)

Formula:
""")
            st.markdown("""
<div class="formula-box">
t = (x̄ − μ₀) / (s / √n)

where:
  x̄  = sample mean
  μ₀ = hypothesised population mean
  s  = sample standard deviation (ddof = 1)
  n  = sample size
  df = n − 1

Cohen's d = (x̄ − μ₀) / s
</div>
""", unsafe_allow_html=True)
            st.markdown("""
**Non-Parametric: Wilcoxon Signed-Rank Test (One-Sample)**

H₀: Population median = μ₀  
Procedure: Compute differences dᵢ = xᵢ − μ₀, rank |dᵢ|, compute W⁺ and W⁻.
Effect size r = Z / √N
""")
        
        with st.expander("Paired (Dependent) Samples Tests", expanded=True):
            st.markdown("""
**Parametric: Paired-Samples t-Test**

H₀: μ_d = 0 (no mean difference)  
H₁: μ_d ≠ 0 (two-sided) | μ_d > 0 | μ_d < 0

Formula:
""")
            st.markdown("""
<div class="formula-box">
t = d̄ / (s_d / √n)

where:
  d̄  = mean of paired differences (dᵢ = x₁ᵢ − x₂ᵢ)
  s_d = standard deviation of differences
  n   = number of pairs
  df  = n − 1

Cohen's d_z = d̄ / s_d
</div>
""", unsafe_allow_html=True)
            st.markdown("""
**Non-Parametric: Wilcoxon Signed-Rank Test (Paired)**

H₀: Distribution of differences is symmetric about zero  
Same procedure as one-sample variant, applied to paired differences.
""")
        
        with st.expander("Independent Samples Tests", expanded=True):
            st.markdown("""
**Parametric: Student's t-Test (equal variances)**
""")
            st.markdown("""
<div class="formula-box">
t = (x̄₁ − x̄₂) / (s_p · √(1/n₁ + 1/n₂))

s_p² = [(n₁−1)s₁² + (n₂−1)s₂²] / (n₁ + n₂ − 2)
df = n₁ + n₂ − 2
</div>
""", unsafe_allow_html=True)
            st.markdown("""
**Parametric: Welch's t-Test (unequal variances)**
""")
            st.markdown("""
<div class="formula-box">
t = (x̄₁ − x̄₂) / √(s₁²/n₁ + s₂²/n₂)

df (Satterthwaite) = (s₁²/n₁ + s₂²/n₂)² / 
                     [(s₁²/n₁)²/(n₁−1) + (s₂²/n₂)²/(n₂−1)]

Cohen's d = (x̄₁ − x̄₂) / s_pooled
</div>
""", unsafe_allow_html=True)
            st.markdown("""
**Non-Parametric: Mann-Whitney U Test**

H₀: The two populations have the same distribution  
Effect size r = Z / √(n₁ + n₂)
""")
        
        st.markdown("### 4. Effect Size Interpretation")
        st.markdown("""
| Magnitude   | Cohen's d | Rank-biserial r |
|-------------|-----------|-----------------|
| Negligible  | < 0.20    | < 0.10          |
| Small       | 0.20–0.49 | 0.10–0.29       |
| Medium      | 0.50–0.79 | 0.30–0.49       |
| Large       | ≥ 0.80    | ≥ 0.50          |

*Based on Cohen (1988) and Rosenthal (1991) conventions.*
""")
        
        st.markdown("### 5. Assumptions & Limitations")
        st.markdown("""
- **Independence**: All tests assume observations are independent within groups.
- **Scale**: t-tests and Wilcoxon tests assume interval-level data; Mann-Whitney U requires at least ordinal data.
- **Sample size**: Very small samples (n < 5) may yield unreliable results regardless of test choice.
- **Ties**: Wilcoxon and Mann-Whitney procedures handle ties via midrank correction.
- **Multiple testing**: This suite performs single comparisons only. Apply appropriate corrections (e.g., Bonferroni) when conducting multiple tests.
""")
        
        st.markdown("### 6. References")
        st.markdown("""
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum.
- Field, A. (2018). *Discovering Statistics Using IBM SPSS Statistics* (5th ed.). SAGE.
- Hollander, M., Wolfe, D.A., & Chicken, E. (2013). *Nonparametric Statistical Methods* (3rd ed.). Wiley.
- Shapiro, S.S., & Wilk, M.B. (1965). An analysis of variance test for normality. *Biometrika*, 52(3–4), 591–611.
- Welch, B.L. (1947). The generalization of "Student's" problem when several different population variances are involved. *Biometrika*, 34(1–2), 28–35.
""")


if __name__ == "__main__":
    main()
