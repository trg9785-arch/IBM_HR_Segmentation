"""Workforce Lens — full Streamlit development code.

Required files in ONE folder: app.py, requirements.txt, IBM_Dataset.csv.
Python 3.12 tested. Install: python -m pip install -r requirements.txt
Run: python -m streamlit run app.py
Open http://localhost:8501 if a browser does not open automatically.

All preprocessing, feature justification, candidate K-Means models, Elbow and
Silhouette evaluation, cluster profiles, simulator, and UI styling are here.
No additional modules, assets, models or precomputed files are required.
Streamlit is the user's revised choice, replacing the original Dash direction.
%run app.py in Colab does not display a Streamlit dashboard; use the server
command above in a local environment or a Streamlit-compatible host.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(page_title='Workforce Lens | IBM HR',page_icon='◉',layout='wide')
NUM=['Age','Education','JobLevel','JobSatisfaction','MonthlyIncome','TotalWorkingYears','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsWithCurrManager']
CAT=['Department','Gender']; FEATURES=NUM+CAT
RANGES=dict(zip(NUM,[(18,60),(1,5),(1,5),(1,4),(1009,19999),(0,40),(1,4),(0,40),(0,18),(0,17)]))
ORD={'Education':['Below College','College','Bachelor','Master','Doctor'],'JobLevel':['Entry','Junior','Mid','Senior','Executive'],'JobSatisfaction':['Low','Medium','High','Very High'],'WorkLifeBalance':['Bad','Good','Better','Best']}
APP_DIR=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
if not (APP_DIR / 'IBM_Dataset.csv').exists():
    st.error('Place IBM_Dataset.csv in the same folder as app.py, then reload.')
    st.stop()
DF=pd.read_csv(APP_DIR / 'IBM_Dataset.csv')
if set(DF.columns)!=set(FEATURES+['Attrition']): raise ValueError('Unexpected CSV schema')
if not DF.Attrition.isin(['Yes','No']).all(): raise ValueError('Invalid Attrition values')

@st.cache_resource(show_spinner="Fitting and evaluating workforce segments…")
def fit_analysis(DF):
    PRE=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(sparse_output=False,handle_unknown='error'))]),CAT)])
    X=PRE.fit_transform(DF[FEATURES]) # Attrition cannot enter this fitted pipeline.
    MODELS={}; rows=[]
    for k in range(2,11):
        model=KMeans(n_clusters=k,random_state=42,n_init=20).fit(X)
        MODELS[k]=model
        rows.append(dict(k=k,inertia=model.inertia_,silhouette=silhouette_score(X,model.labels_),minimum_size=int(np.bincount(model.labels_).min())))
    EVAL=pd.DataFrame(rows)
    # Near ties within .01 of the best silhouette resolve to the smallest k.
    RECOMMENDED=int(EVAL.loc[EVAL.silhouette>=EVAL.silhouette.max()-.01,'k'].min())
    MODEL=MODELS[RECOMMENDED]
    # Stable neutral numbering by feature-derived mean income only; no outcome ordering.
    ORDER=sorted(range(RECOMMENDED),key=lambda c:DF.loc[MODEL.labels_==c,'MonthlyIncome'].mean())
    LABELS={c:f'Segment {i+1} · '+('Lower-income career stage' if i==0 else 'Higher-income career stage' if i==RECOMMENDED-1 else 'Intermediate career stage') for i,c in enumerate(ORDER)}
    Y=(DF.Attrition=='Yes').astype(int); OVERALL=Y.mean()
    DATA=DF.assign(cluster=MODEL.labels_,segment=[LABELS[c] for c in MODEL.labels_],left=Y)
    STABILITY=pd.DataFrame([{'seed':s,'ARI':adjusted_rand_score(MODEL.labels_,KMeans(n_clusters=RECOMMENDED,random_state=s,n_init=20).fit_predict(X))} for s in [0,7,21,99,2026]])
    DEFAULT={f:int(DF[f].median()) for f in NUM}; DEFAULT.update({f:DF[f].mode()[0] for f in CAT})
    PRESET={**DEFAULT,'JobLevel':2,'JobSatisfaction':1,'YearsInCurrentRole':3}
    PRESET['YearsAtCompany']=max(PRESET['YearsAtCompany'],3); PRESET['TotalWorkingYears']=max(PRESET['TotalWorkingYears'],PRESET['YearsAtCompany'])
    TRAIN_DIST=np.linalg.norm(X-MODEL.cluster_centers_[MODEL.labels_],axis=1)
    DIST95=float(np.quantile(TRAIN_DIST,.95))
    GROUPS={f:[i] for i,f in enumerate(NUM)}; start=len(NUM)
    for f,categories in zip(CAT,PRE.named_transformers_['category'].named_steps['encode'].categories_):
        GROUPS[f]=list(range(start,start+len(categories))); start+=len(categories)

    return {'PRE': PRE, 'X': X, 'MODELS': MODELS, 'EVAL': EVAL, 'RECOMMENDED': RECOMMENDED, 'MODEL': MODEL, 'ORDER': ORDER, 'LABELS': LABELS, 'Y': Y, 'OVERALL': OVERALL, 'DATA': DATA, 'STABILITY': STABILITY, 'DEFAULT': DEFAULT, 'PRESET': PRESET, 'TRAIN_DIST': TRAIN_DIST, 'DIST95': DIST95, 'GROUPS': GROUPS}

globals().update(fit_analysis(DF))

def wilson(s,n):
    if not n:return (np.nan,np.nan)
    z=1.95996398454;p=s/n;d=1+z*z/n
    mid=(p+z*z/(2*n))/d;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return mid-half,mid+half

def profiles():
    out=[]
    for c in ORDER:
        g=DATA[DATA.cluster==c];n=len(g);s=int(g.left.sum());lo,hi=wilson(s,n)
        out.append({'cluster':c,'Segment':LABELS[c],'Employees':n,'Sample %':100*n/len(DATA),'Leavers':s,'Rate %':100*s/n,'95% low %':100*lo,'95% high %':100*hi,'Difference pp':100*(s/n-OVERALL),'Share of all leavers %':100*s/Y.sum(),**g[NUM].mean().to_dict()})
    return pd.DataFrame(out)
PROFILES=profiles()
AUDIT=pd.DataFrame([{'Field':f,'dtype':str(DF[f].dtype),'Missing':int(DF[f].isna().sum()),'Distinct':DF[f].nunique(),'Range / categories':f'{DF[f].min()} – {DF[f].max()}' if f in NUM else ', '.join(sorted(DF[f].unique()))} for f in DF])
VIOLATIONS={f'{a} > {b}':int((DF[a]>DF[b]).sum()) for a,b in [('YearsInCurrentRole','YearsAtCompany'),('YearsWithCurrManager','YearsAtCompany'),('YearsAtCompany','TotalWorkingYears')]}
DISCREPANCY='The prompt claims 13 non-Attrition variables, but its own list and this CSV contain 12 characteristics plus Attrition (13 columns total). All 12 available characteristics are used: 10 numeric/ordinal and 2 categorical. No thirteenth characteristic is invented. OverTime is absent.'
ETHICS='Attrition combines voluntary and involuntary departures, which cannot be distinguished. Cross-sectional associations do not establish causes. Segment membership does not predict individual departure, and observed rates may not generalize. Demographics are included only because the assignment requires them. Use for supportive retention planning only; never for hiring, promotion, discipline or adverse employment actions. Low satisfaction or demographic membership must not trigger punitive treatment.'
METHOD=f'''Preprocessing is fitted once on all {len(DF):,} rows, using only FEATURES. Median imputation and StandardScaler apply to {', '.join(NUM)}. Most-frequent imputation and full one-hot encoding apply to Department and Gender. No dummy standardization or block weighting is used (weights 1). The {len(FEATURES)} original inputs become {X.shape[1]} transformed columns. Scaling prevents dollars and years from dominating through units. Ordinal scores assume equal spacing; one-hot coordinates permit Euclidean distance but categorical means are proportions. Correlated age, income, job level and tenure can overweight career stage. K-Means favors compact Euclidean clusters and may fit mixed workforce data imperfectly. Centroids are means in encoded space; original-unit profile means summarize actual assigned employees.'''
RATIONALE=f'Choose the smallest k within 0.01 of the maximum silhouette: recommended k={RECOMMENDED}. All candidates use random_state=42 and n_init=20. Inertia decreases mechanically with k; the elbow is ambiguous and is not treated as proof. Minimum sizes and feature profiles are shown for review. Initialization stability supports reproducibility, not real-world validity. Attrition is never used in selection. The selected partition is fixed throughout the application.'
FRAMEWORK={
    'Problem':'Advise Dr. Sarah Chen and HR planning teams in the assigned 2025 IBM classroom scenario. Answer questions a–e and identify supportive retention priorities. Do not score individuals or make employment decisions.',
    'Data':DISCREPANCY+' '+METHOD,
    'Insights':'Compare evaluated feature-based segments, then describe their observed attrition. Separate associations, interpretations, and proposed interventions requiring evaluation.',
    'Deployment':'Streamlit dashboard selected by the team in a later revision. This replaces the original Dash/Plotly Cloud direction; it is not the original Option 2 implementation. Keep app.py, requirements.txt and IBM_Dataset.csv together. Install: python -m pip install -r requirements.txt. Run: python -m streamlit run app.py. No chatbot, API key or external data call is needed. No hosted deployment has been performed.'
}

def compare(field):
    return DATA.groupby(field,observed=True).agg(Employees=('left','size'),Leavers=('left','sum'),Rate=('left','mean')).reset_index().assign(**{'Rate %':lambda t:100*t.Rate}).drop(columns='Rate')

def questions():
    means=DATA.groupby('Attrition')[NUM].mean();hi=PROFILES.loc[PROFILES['Rate %'].idxmax()];lo=PROFILES.loc[PROFILES['Rate %'].idxmin()]
    texts=[]
    texts.append(('a. Who leaves versus remains?',f"Leavers: {Y.sum()}/{len(DF)} ({100*OVERALL:.2f}%). Mean age {means.loc['Yes','Age']:.1f} vs {means.loc['No','Age']:.1f}; income ${means.loc['Yes','MonthlyIncome']:,.0f} vs ${means.loc['No','MonthlyIncome']:,.0f}. {hi['Segment']} has {int(hi.Leavers)}/{int(hi.Employees)} leavers ({hi['Rate %']:.1f}%) versus {lo['Rate %']:.1f}% in {lo['Segment']}. These are observed group differences."))
    def shares(row,f,value):
        g=DATA[DATA.cluster==row.cluster];return 100*(g[f]==value).mean()
    texts.append(('b. Gender and education',f"Higher-rate segment: female {shares(hi,'Gender','Female'):.1f}%, Bachelor {shares(hi,'Education',3):.1f}%; lower-rate segment: female {shares(lo,'Gender','Female'):.1f}%, Bachelor {shares(lo,'Education',3):.1f}%. Composition does not establish demographic effects; see complete distributions and category-rate denominators in the app."))
    texts.append(('c. Department and job level',f"Higher-rate segment: Sales {shares(hi,'Department','Sales'):.1f}%, entry level {shares(hi,'JobLevel',1):.1f}%; lower-rate segment: Sales {shares(lo,'Department','Sales'):.1f}%, entry level {shares(lo,'JobLevel',1):.1f}%. Department and job-level category rates in the app are descriptive, not adjusted effects."))
    tenure=['TotalWorkingYears','YearsAtCompany','YearsInCurrentRole','YearsWithCurrManager']
    texts.append(('d. Experience and tenure','Leaver / stayer means (years): '+ '; '.join(f'{f}: {means.loc["Yes",f]:.2f} / {means.loc["No",f]:.2f}' for f in tenure)+'. Segment means: '+ '; '.join(f'{r.Segment}: '+', '.join(f'{f}={r[f]:.1f}' for f in tenure) for _,r in PROFILES.iterrows())+'. These are cross-sectional comparisons, not tenure trends.'))
    texts.append(('e. Satisfaction and work-life balance',f"Leaver / stayer means: satisfaction {means.loc['Yes','JobSatisfaction']:.2f} / {means.loc['No','JobSatisfaction']:.2f}; work-life balance {means.loc['Yes','WorkLifeBalance']:.2f} / {means.loc['No','WorkLifeBalance']:.2f}. Higher-/lower-rate segment satisfaction means {hi.JobSatisfaction:.2f}/{lo.JobSatisfaction:.2f}, balance {hi.WorkLifeBalance:.2f}/{lo.WorkLifeBalance:.2f}. Distributions in the app avoid interpreting ordinal averages as causal effects."))
    return texts

QUESTIONS=questions()
def priority(c):
    r=PROFILES[PROFILES.cluster==c].iloc[0]
    return ('Interpretation: lower income and shorter experience suggest onboarding and career-development needs. Proposal: pilot mentoring, transparent progression and supportive manager check-ins.' if c==ORDER[0] else 'Interpretation: greater income and experience suggest established-career needs. Proposal: pilot career mobility, knowledge-sharing and sustainable workload reviews.')+' Evaluate participation, satisfaction and subsequent aggregate retention with an appropriate comparison design before expansion; no proven savings are claimed.'

def validate(values):
    errors=[]
    for f in NUM:
        v=values.get(f);lo,hi=RANGES[f]
        if not isinstance(v,(int,float,np.number)) or not np.isfinite(v) or int(v)!=v or not lo<=v<=hi:errors.append(f'{f}: enter an integer from {lo} to {hi}.')
    for f in CAT:
        if values.get(f) not in DF[f].unique():errors.append(f'{f}: choose an available category.')
    if errors:return errors
    for a,b in [('YearsInCurrentRole','YearsAtCompany'),('YearsWithCurrManager','YearsAtCompany'),('YearsAtCompany','TotalWorkingYears')]:
        if values[a]>values[b]:errors.append(f'{a} must be <= {b}.')
    return errors

def explain(values):
    errors=validate(values)
    if errors:raise ValueError(' '.join(errors))
    z=PRE.transform(pd.DataFrame([values],columns=FEATURES))[0]
    sq=(MODEL.cluster_centers_-z)**2;d=sq.sum(axis=1);order=np.argsort(d);a,b=map(int,order[:2])
    contributions=pd.DataFrame([{'Variable':f,'Assigned squared contribution':sq[a,idx].sum(),'Runner-up squared contribution':sq[b,idx].sum(),'Runner-up minus assigned':sq[b,idx].sum()-sq[a,idx].sum()} for f,idx in GROUPS.items()])
    assert np.isclose(contributions['Assigned squared contribution'].sum(),d[a])
    assert np.isclose(contributions['Runner-up minus assigned'].sum(),d[b]-d[a])
    return a,b,np.sqrt(d),contributions


# ----- Streamlit presentation: fixed recommended model; display filters only -----
NOTICE='This case is assigned to a workforce segment. The segment’s observed attrition rate is not this employee’s probability of leaving.'
RATE_COLUMNS=['Segment','Employees','Sample %','Leavers','Rate %','95% low %','95% high %','Difference pp','Share of all leavers %']

def table(frame):
    st.dataframe(frame.round(3),hide_index=True,width='stretch')

def graph(fig,key):
    fig.update_layout(template='plotly_white',font={'family':'Arial'},margin=dict(l=35,r=20,t=65,b=45),colorway=['#087f8c','#db893b','#6e60a5'])
    st.plotly_chart(fig,width='stretch',key=key)

def composition(g,f):
    # Reindex to include zero-count categories and preserve ordinal ordering.
    categories=list(range(1,len(ORD[f])+1)) if f in ORD else sorted(DF[f].dropna().unique())
    counts=g[f].value_counts().reindex(categories,fill_value=0)
    t=counts.rename_axis(f).reset_index(name='Employees')
    t['Within segment %']=100*t.Employees/len(g) if len(g) else 0.0
    if f in ORD:t['Meaning']=t[f].map(lambda v:ORD[f][v-1])
    return t

def display_filter(depts):
    g=DATA[DATA.Department.isin(depts or [])]
    if g.empty:return pd.DataFrame(columns=['Segment','Employees','Leavers','Rate %'])
    t=g.groupby('segment',observed=True).agg(Employees=('left','size'),Leavers=('left','sum'),Rate=('left','mean')).reset_index().rename(columns={'segment':'Segment'})
    t['Rate %']=100*t.pop('Rate')
    return t

def set_scenario(source):
    for f in FEATURES:st.session_state['input_'+f]=source[f]
    # Reset and preset immediately refresh both simulator and explanation.
    st.session_state['evaluated_case']=dict(source)
    st.session_state.pop('scenario_error',None)

def render_scenario(values):
    a,b,dist,contrib=explain(values)
    r=PROFILES[PROFILES.cluster==a].iloc[0]
    st.subheader(LABELS[a])
    cols=st.columns(3)
    cols[0].metric('Segment employees',f'{int(r.Employees):,}')
    cols[1].metric('Observed segment attrition',f'{r["Rate %"]:.2f}%',f'{r["Difference pp"]:+.2f} pp vs sample',delta_color='off')
    cols[2].metric('Observed leavers / segment size',f'{int(r.Leavers)} / {int(r.Employees)}')
    st.caption(f'Wilson 95% interval: {r["95% low %"]:.2f}%–{r["95% high %"]:.2f}%. This interval omits clustering and unknown sampling uncertainty.')
    st.write(priority(a))
    means=DATA[DATA.cluster==a][NUM].mean()
    table(pd.DataFrame({'Variable':NUM,'Scenario':[values[f] for f in NUM],'Segment mean':means.values,'Scenario minus mean':[values[f]-means[f] for f in NUM]}))
    st.write('; '.join(f'{f}: case {values[f]}, segment mode {DATA[DATA.cluster==a][f].mode()[0]}' for f in CAT))
    st.info(NOTICE)
    return a,b,dist,contrib

st.markdown('''<style>
.stApp{background:#f3f6f9;color:#172f45}
[data-testid="stMetric"]{background:white;border-top:4px solid #087f8c;padding:18px;border-radius:10px}
h1,h2,h3{color:#153c55} [data-testid="stSidebar"]{background:#e9f1f5}
</style>''',unsafe_allow_html=True)
st.title('Workforce Lens')
st.caption('IBM HR classroom consulting project · Streamlit dashboard · supportive retention planning')
st.info(NOTICE)
with st.expander('Dataset discrepancy and revised deployment choice',expanded=False):
    st.write(DISCREPANCY)
    st.write(FRAMEWORK['Deployment'])
with st.sidebar:
    st.header('Analysis reference')
    st.metric('Recommended and selected k',RECOMMENDED)
    st.caption('Fixed model throughout. Display filters and simulated cases never retrain it.')
    st.write(f'{len(DF):,} employees · {int(Y.sum())} leavers · {OVERALL:.2%} observed attrition')
    st.caption('To run: python -m streamlit run app.py')

names=['Executive Overview','Cluster Quality','Employee Segments','Business Questions','Scenario Simulator','Interpretability','Explainability','Framework and Limitations']
tabs=st.tabs(names)

with tabs[0]:
    st.header('Executive Overview')
    st.write('Advise Dr. Sarah Chen in the assigned 2025 IBM classroom scenario on supportive workforce retention priorities. The file’s provenance and population representativeness are not independently verified.')
    cols=st.columns(5)
    for col,label,value in zip(cols,['Employees','Leavers','Observed attrition','Selected k','Silhouette'],[f'{len(DF):,}',str(int(Y.sum())),f'{OVERALL:.2%}',str(RECOMMENDED),f'{EVAL.set_index("k").loc[RECOMMENDED,"silhouette"]:.3f}']):col.metric(label,value)
    graph(px.bar(PROFILES,x='Segment',y='Employees',title='Full sample: employees per segment'),'overview_size')
    ratefig=go.Figure(go.Bar(x=PROFILES.Segment,y=PROFILES['Rate %'],error_y=dict(type='data',array=PROFILES['95% high %']-PROFILES['Rate %'],arrayminus=PROFILES['Rate %']-PROFILES['95% low %']),text=[f'{s}/{n}' for s,n in zip(PROFILES.Leavers,PROFILES.Employees)]))
    ratefig.add_hline(y=100*OVERALL,line_dash='dash',annotation_text='Sample overall rate')
    ratefig.update_layout(title='Observed attrition: leavers / employees in each segment',yaxis_title='Observed attrition % · Wilson 95% intervals')
    graph(ratefig,'overview_rates');table(PROFILES[RATE_COLUMNS])
    st.caption(f'Rate denominator: employees within each segment. Share of all leavers uses all {int(Y.sum())} leavers. Wilson intervals do not account for all sampling, selection or clustering uncertainty.')
    st.subheader('Evidence-based findings')
    st.write(QUESTIONS[0][1]);st.write(QUESTIONS[4][1])

with tabs[1]:
    st.header('Cluster Quality')
    st.write(RATIONALE)
    c1,c2=st.columns(2)
    with c1:graph(px.line(EVAL,x='k',y='inertia',markers=True,title='Elbow inspection · k=2–10'),'elbow')
    with c2:graph(px.line(EVAL,x='k',y='silhouette',markers=True,title='Silhouette · full transformed feature space'),'silhouette')
    table(EVAL)
    st.subheader('Selected solution: full-sample cluster sizes');table(PROFILES[['Segment','Employees','Sample %']])
    st.subheader('Initialization stability at recommended k');table(STABILITY)
    st.write(f'Five supplementary seeds: mean adjusted Rand index {STABILITY.ARI.mean():.4f}, range {STABILITY.ARI.min():.4f}–{STABILITY.ARI.max():.4f}. ARI is label-invariant. These checks measure initialization stability, not real-world validity.')
    st.warning('Separation is modest. K-Means imposes a compact Euclidean partition; silhouette and elbow results do not prove natural groups, causal structure or predictive ability.')

with tabs[2]:
    st.header('Employee Segments')
    c=st.selectbox('Choose segment',ORDER,format_func=lambda value:LABELS[value],key='selected_segment')
    r=PROFILES[PROFILES.cluster==c];g=DATA[DATA.cluster==c]
    table(r[RATE_COLUMNS]);st.write(priority(c))
    st.subheader('Original-unit mean profiles, with the full-sample reference')
    orig=PROFILES.set_index('Segment')[NUM].copy();orig.loc['Sample overall']=DF[NUM].mean()
    table(orig.reset_index())
    z=PROFILES.set_index('Segment')[NUM].sub(DF[NUM].mean()).div(DF[NUM].std(ddof=0))
    graph(px.imshow(z,aspect='auto',color_continuous_scale='RdBu_r',zmin=-2,zmax=2,title='Mean numeric profiles relative to sample · standard deviations'),'profile_heatmap')
    st.subheader('Composition within selected segment')
    for f in CAT+list(ORD):
        with st.expander(f,expanded=f in CAT):
            t=composition(g,f);table(t)
            graph(px.bar(t,x=f,y='Within segment %',hover_data=['Employees'],title=f'{f}: composition of {LABELS[c]}'),'composition_'+f)
    st.subheader('Department filter for displayed rates only')
    depts=st.multiselect('Displayed departments',sorted(DF.Department.unique()),default=sorted(DF.Department.unique()),key='department_filter')
    denominator=int(DATA.Department.isin(depts).sum())
    st.caption(f'Filtered denominator: {denominator} / {len(DATA)} full-reference employees. Full-reference segment metrics remain above. Segment memberships and models never change.')
    ft=display_filter(depts)
    if ft.empty:st.info('No employees selected. Choose at least one department to display filtered statistics.')
    else:table(ft)

with tabs[3]:
    st.header('Business Questions')
    for title,answer in QUESTIONS:st.subheader(title);st.write(answer)
    st.subheader('Leaver versus stayer comparisons · original-unit means');table(DATA.groupby('Attrition')[NUM].mean().reset_index())
    st.caption('Category-rate denominators below are all employees in the category, unlike within-segment composition percentages. These are unadjusted descriptive comparisons.')
    for f in CAT+list(ORD):
        with st.expander(f+' · within-category attrition',expanded=f=='Department'):
            t=compare(f);table(t)
            graph(px.bar(t,x=f,y='Rate %',hover_data=['Employees','Leavers'],title=f'{f}: observed leavers / employees in category'),'category_rates_'+f)
    st.write('Observed associations: the figures describe this cross-sectional sample. Interpretations: career-stage patterns may suggest different support needs. Proposed interventions: voluntary supportive pilots, evaluated with appropriate comparison groups; no causal effect or proven financial return is claimed.')

# Widgets use independent edit state; only explicit evaluation commits a case.
for f in FEATURES:
    if 'input_'+f not in st.session_state:st.session_state['input_'+f]=DEFAULT[f]
if 'evaluated_case' not in st.session_state:st.session_state['evaluated_case']=dict(DEFAULT)
with tabs[4]:
    st.header('Scenario Simulator')
    st.write('All 12 available inputs are included. Defaults use valid integer medians and category modes. The instructor preset uses job level 2, three years in the current role and low satisfaction. OverTime is absent, so overtime cannot be evaluated.')
    b1,b2=st.columns(2)
    b1.button('Reset to defaults',on_click=set_scenario,args=(DEFAULT,),key='reset')
    b2.button('Instructor preset',on_click=set_scenario,args=(PRESET,),key='preset')
    with st.form('employee_case'):
        cols=st.columns(3)
        for i,f in enumerate(FEATURES):
            with cols[i%3]:
                if f in CAT:st.selectbox(f,sorted(DF[f].unique()),key='input_'+f)
                elif f in ORD:st.selectbox(f,list(range(1,len(ORD[f])+1)),format_func=lambda v,f=f:f'{v}: {ORD[f][v-1]}',key='input_'+f)
                else:st.number_input(f,min_value=RANGES[f][0],max_value=RANGES[f][1],step=1,key='input_'+f)
        submitted=st.form_submit_button('Evaluate scenario')
    if submitted:
        case={f:st.session_state['input_'+f] for f in FEATURES}
        errors=validate(case)
        if errors:st.session_state['scenario_error']=' '.join(errors)
        else:
            st.session_state['evaluated_case']=case
            st.session_state.pop('scenario_error',None)
    if 'scenario_error' in st.session_state:
        st.error(st.session_state['scenario_error'])
        st.caption('Invalid case rejected. The last valid evaluated case is retained below and in Explainability.')
    case=st.session_state['evaluated_case']
    if any(st.session_state['input_'+f]!=case[f] for f in FEATURES):st.caption('Results show the last valid evaluated case. Submit valid changes to update both tabs.')
    render_scenario(case)

with tabs[5]:
    st.header('Interpretability')
    st.write(METHOD)
    st.write('K-Means initializes centers with seeded k-means++, assigns each transformed employee to its nearest centroid, recomputes centroid means, and repeats until convergence. Settings: random_state=42, n_init=20, max_iter=300, tol=0.0001. For each k, the lowest-inertia initialization is retained. Candidate models are cached once for the dataset and reused on interactions.')
    st.write(RATIONALE)
    st.write('Attrition is excluded from preprocessing, centroid fitting, all distances, k selection, label ordering and scenario assignment. It is joined only after fitting to describe observed group outcomes. Labels use ascending mean income, not observed attrition. There are no classifiers, individual departure probabilities or classification metrics.')
    table(pd.DataFrame({'Original variable':FEATURES,'Treatment':['Median imputation + standardization' if f in NUM else 'Mode imputation + full one-hot encoding' for f in FEATURES],'Transformed columns':[len(GROUPS[f]) for f in FEATURES]}))
    st.caption('All available characteristics are used as required; there is no outcome-driven feature selection. Inclusion of Gender and other demographics follows the assignment, not an employment-decision recommendation.')

with tabs[6]:
    st.header('Explainability')
    st.write('This tab uses the last valid evaluated case from Scenario Simulator. Each distance is computed in the actual full transformed space. Large squared contributions indicate mismatch with a centroid. They are not attrition effects or causal drivers. Distances are not probabilities or calibrated confidence.')
    case=st.session_state['evaluated_case'];a,b,dist,contrib=explain(case)
    st.caption('Evaluated case: '+', '.join(f'{f}={case[f]}' for f in FEATURES))
    dt=pd.DataFrame({'Segment':[LABELS[c] for c in range(RECOMMENDED)],'Distance':dist,'Assignment':['Assigned' if c==a else 'Runner-up' if c==b else 'Other' for c in range(RECOMMENDED)]})
    graph(px.bar(dt,x='Segment',y='Distance',color='Assignment',title='Euclidean distance to every centroid'),'case_distances');table(dt)
    st.subheader('Per-original-variable contributions');table(contrib)
    graph(px.bar(contrib,x='Runner-up minus assigned',y='Variable',orientation='h',title='Positive differences support assigned segment over runner-up'),'case_margin')
    st.write('Contributions from all Department dummy columns are summed into Department; Gender dummy columns are summed into Gender. Positive runner-up minus assigned values favor the assigned segment relative to the runner-up; negative values favor the runner-up on that variable.')
    st.code(f'Assigned contribution sum = {contrib["Assigned squared contribution"].sum():.10f}\nAssigned squared distance = {dist[a]**2:.10f}\nMargin contribution sum = {contrib["Runner-up minus assigned"].sum():.10f}\nRunner-up² − assigned² = {dist[b]**2-dist[a]**2:.10f}')
    st.caption('Both numerical identities are asserted in explain().')
    distance_message=f'Empirical pooled 95th percentile of observed nearest-centroid distances = {DIST95:.3f}; case distance = {dist[a]:.3f}. '
    if dist[a]>DIST95:st.warning(distance_message+'This case is unusually distant relative to that fitted-data reference.')
    else:st.info(distance_message+'This case does not exceed that reference.')
    st.caption('This transparent threshold uses the fitted sample. It is not a probability, a coverage guarantee or evidence that the case resembles future employees.')

with tabs[7]:
    st.header('Framework and Limitations')
    st.write('Operational interpretation of Problem–Data–Insights–Deployment. The class’s exact template was not supplied.')
    for title,body in FRAMEWORK.items():st.subheader(title);st.write(body)
    st.subheader('Actual dataset audit')
    st.write(f'{len(DF):,} rows; {len(DF.columns)} total columns; {int(DF.isna().sum().sum())} missing values; {int(DF.duplicated().sum())} identical full rows, retained; constant columns: {[c for c in DF if DF[c].nunique()==1] or "none"}. No identifier exists, so identical characteristics would not establish duplicate employees.')
    table(AUDIT)
    range_audit=pd.DataFrame([{'Field':f,'Allowed minimum':RANGES[f][0],'Allowed maximum':RANGES[f][1],'Out-of-range source rows':int((DF[f].notna() & ~DF[f].between(*RANGES[f])).sum()),'Non-integer source rows':int((DF[f].notna() & (DF[f]%1!=0)).sum())} for f in NUM])
    table(range_audit)
    st.write('Source-data relationship violations (audited, not deleted or modified): '+str(VIOLATIONS))
    st.write('Simulation validation differs from source auditing: user-entered cases must meet allowed ranges, integer types, known categories and tenure relationships before assignment. Source observations are reported as supplied. No independent verification of actual IBM 2025 records or whole-workforce representativeness is claimed.')
    st.warning(ETHICS)
    st.subheader('Run and reproduce')
    st.code('python -m pip install -r requirements.txt\npython -m streamlit run app.py',language='bash')
    st.caption('Keep app.py, requirements.txt and IBM_Dataset.csv together. Python 3.12 tested. Streamlit normally opens http://localhost:8501. All code and styling are in app.py; no extra files or external API are needed. No hosting deployment is claimed.')
    st.download_button('Download aggregate profiles',PROFILES.to_csv(index=False),file_name='aggregate_profiles.csv',mime='text/csv')
    st.subheader('Team contribution statement')
    st.write('TEAM TO COMPLETE HONESTLY: list only actual contributors and their actual work. Do not assume equal participation. AI assistance generated and revised the application and report; the team must review, verify and explain the submitted work.')

st.caption('Supportive planning only. Demographic membership or low satisfaction must never trigger punitive treatment.')
