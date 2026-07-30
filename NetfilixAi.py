import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ تنظیمات صفحه ------------------

st.set_page_config(
    page_title="Netflix AI Movie",
    page_icon="🎬",
    layout="wide"
)

# ------------------ CSS ------------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#111827,#1f2937,#0f172a);
color:white;
}

h1{
text-align:center;
color:#ff4b4b;
}

.main-title{
font-size:55px;
font-weight:bold;
text-align:center;
margin-bottom:10px;
}

.sub-title{
text-align:center;
font-size:22px;
color:#dddddd;
margin-bottom:30px;
}

div.stButton > button{
background:#ff4b4b;
color:white;
border-radius:15px;
height:60px;
font-size:22px;
font-weight:bold;
width:100%;
transition:0.3s;
}

div.stButton > button:hover{
background:#ff2222;
transform:scale(1.03);
}

.movie-card{

background:rgba(255,255,255,0.08);

padding:25px;

border-radius:20px;

margin-bottom:25px;

box-shadow:0px 8px 20px rgba(0,0,0,.35);

border:1px solid rgba(255,255,255,.15);

}

.footer{

text-align:center;

margin-top:50px;

color:gray;

font-size:15px;

}

</style>
""",unsafe_allow_html=True)

# ------------------ عنوان ------------------

st.markdown(
"""
<div class="main-title">
🎬 Netflix AI Movie Finder
</div>

<div class="sub-title">
🤖 Describe the movies you enjoy and let AI recommend the perfect ones.
</div>
""",
unsafe_allow_html=True
)

# ------------------ مدل ------------------

with open("model_Net.pkl","rb") as file:

    model=pickle.load(file)

    data_number=pickle.load(file)

    data=pickle.load(file)

# ------------------ ورودی ------------------

que=st.text_input(
"🎥 Describe a movie you like",
placeholder="Example : I love emotional action movies with a lot of suspense..."
)

# ------------------ دکمه ------------------

if st.button("🎯 Find Movies"):

    if que.strip()=="":

        st.warning("Please enter a description.")

    else:

        with st.spinner("Searching for the best movies... 🔍"):

            question_vector=model.transform([que])

            similarity=cosine_similarity(
                question_vector,
                data_number
            ).flatten()

            top3=similarity.argsort()[::-1][:3]

        st.success("✨ Here are your recommendations!")

        for i in top3:

            percent=float(similarity[i])*100

            st.markdown(
            f"""
            <div class="movie-card">

            <h2>🎬 {data.iloc[i]["title"]}</h2>

            <p><b>⭐ IMDb Score:</b> {data.iloc[i]["imdb_score"]}</p>

            <p><b>📅 Release Year:</b> {data.iloc[i]["release_year"]}</p>

            <p><b>📝 Description:</b></p>

            <p>{data.iloc[i]["description"]}</p>

            </div>

            """,
            unsafe_allow_html=True
            )

            st.progress(min(percent/100,1.0))

            st.write(f"🎯 Similarity : **{percent:.1f}%**")

st.markdown("""
<div class="footer">

Made with ❤️ using Python + Streamlit

</div>
""",unsafe_allow_html=True)
