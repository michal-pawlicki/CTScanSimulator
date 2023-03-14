import streamlit as st


def main():
    st.title('CT scan simulator')
    st.subheader('by Agnieszka Grzymaska and Michał Pawlicki')
    st.markdown('---')
    side_bar()


def side_bar():
    st.sidebar.markdown('# Set scanner options')
    st.sidebar.markdown(""" $\Delta$ $\\alpha$ $value$""")
    st.sidebar.select_slider("Select value of alpha", options=range(1, 20, 2))
    st.sidebar.markdown("""$Number$ $of$ $detectors$""")
    st.sidebar.select_slider("Select value of n", options=range(1, 20, 2))
    st.sidebar.markdown("""$Span$ $of$ $the$ $emitter$ $system$""")
    st.sidebar.select_slider("Select value of l", options=range(1, 20, 2))


if __name__ == '__main__':
    main()
