import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
from collections import Counter
import numpy as np
import matplotlib.pyplot as mlt
from scores import scores_function
from iris import iris_prediction
from matplotlib.backends.backend_agg import RendererAgg

st.set_option('deprecation.showPyplotGlobalUse', False)

st.sidebar.write("**Select the App**")
app_type = st.sidebar.selectbox(" ", [
                                "Score Prediction", "Iris Prediction", "IPL Visualization"], 2)

if app_type == "Score Prediction":
    scores_function()

elif app_type == "Iris Prediction":
    iris_prediction()

else:

    @st.cache
    def load_data(path):
        data = pd.read_csv(path)
        data.replace(['Mumbai Indians', 'Kolkata Knight Riders', 'Royal Challengers Bangalore', 'Deccan Chargers', 'Chennai Super Kings',
                      'Rajasthan Royals', 'Delhi Daredevils', 'Gujarat Lions', 'Kings XI Punjab',
                      'Sunrisers Hyderabad', 'Rising Pune Supergiants', 'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiant'], ['MI', 'KKR', 'RCB', 'DC', 'CSK', 'RR', 'DD', 'GL', 'KXIP', 'SRH', 'RPS', 'KTK', 'PW', 'RPS'], inplace=True)

        return data

    matches = load_data('matches.csv')
    deliveries = load_data('deliveries.csv')

    st.sidebar.write("** Analysis **")

    analysis = st.sidebar.radio(
        "Select the Category", ("Home", "General", "Batsmen", "Bowlers",
                                "Head to Head Records", "Toss"), 0
    )

    # Home Page

    if analysis == "Home":
        st.title("Data Analysis of Indian Premier League")
        st.write("Shape of Matches Dataframe is {}".format(matches.shape))
        st.write("Showing Random 100 rows of Matches.")
        st.write(matches.sample(100))

        st.write("Shape of deliveries Dataframe is {}".format(deliveries.shape))
        st.write("Showing Random 100 rows of deliveries.")
        st.write(deliveries.sample(100))

    # Head to Head data

    if analysis == "Head to Head Records":
        st.title("Analysis of Head to Head Records")
        col1, col2 = st.beta_columns(2)

        def plot_head_to_head(t1, t2):
            try:
                head_to_head = matches[((matches.team1 == t1) & (matches.team2 == t2))
                                       | ((matches.team2 == t1) & (matches.team1 == t2))].sort_values(['season'])
                if head_to_head is not None:
                    sns.set_style("dark")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax = sns.countplot(x="season", hue="winner",
                                       data=head_to_head, palette="summer")
                    st.pyplot(fig)
            except ValueError:
                st.error("No Data Available")

        with col1:
            st.write("**Head to Head Records of Teams**")

            teams = ['MI', 'KXIP', 'CSK', 'RCB', 'KKR', 'DD', 'RR', 'RR', 'SRH', 'DC',
                     'PW', 'RPS', 'GL', 'KTK']

            team1 = st.selectbox("Select team 1", teams, 0)

            team2 = st.selectbox("Select team 2", teams, 3)

            if (team1 != team2):
                plot_head_to_head(team1, team2)
                st.write(
                    "*DD is for Delhi Daredevils + Delhi Capitals and DC stands for Deccan Chargers.*")

            else:
                st.error("Select Different Teams")

# Individual Teams Performance
            with col2:
                st.write("**Performance of a Specific Team in a Season**")

                ls = list(np.arange(2008, 2020, 1))
                season = st.selectbox("Select the Season", ls, 11)
                all_teams_in_the_season = list(matches[matches.season == season].team1.value_counts(
                ).rename_axis('teams').reset_index(name='counts')["teams"].explode().unique())
                team = st.selectbox(
                    "Select a team", all_teams_in_the_season, 0)

                s = matches[(matches.season == season) & (
                    (matches.team1 == team) | (matches.team2 == team))]

                for i in range(s.shape[0]):
                    if (s.team1.iloc[i] != team):
                        s.team2.iloc[i] = s.team1.iloc[i]
                        s.team1.iloc[i] = team
                    if (s.winner.iloc[i] != team):
                        s.winner.iloc[i] = "Other"

                fig, ax = plt.subplots(figsize=(10, 6))
                ax = sns.countplot(x="team2", hue="winner",
                                   data=s, palette="winter")
                plt.xlabel("Team 2")
                plt.ylabel("Head to Head Count")

                plt.title("Performance of {} in {} Season".format(team, season))
                st.pyplot(fig)

    # Toss Analysis

    if analysis == "Toss":
        st.title("Analysis of Toss")
        col1, col2 = st.beta_columns(2)
        col3, col4 = st.beta_columns(2)

        with col1:
            st.write("**Decisions to field or bat per season**")
            import matplotlib.pyplot as mlt
            fig, ax = mlt.subplots(figsize=(10, 6))
            ax = sns.countplot(x='season', hue='toss_decision',
                               data=matches, palette=["#FF0000", "#FF6347"])
            st.pyplot(fig)

        with col2:
            st.write("**Total Tosses Won by a Team**")
            fig, ax = mlt.subplots(figsize=(10, 6))
            ax = matches['toss_winner'].value_counts().plot.bar(
                width=0.9, color=sns.color_palette('RdYlGn', 20))
            for p in ax.patches:
                ax.annotate(format(p.get_height()),
                            (p.get_x() + 0.15, p.get_height() + 1))
            st.pyplot(fig)

        with col4:
            st.write("")

        with col3:
            st.write("**Is Toss Winner Also the Match Winner?**")
            df = matches[matches['toss_winner'] == matches['winner']]
            slices = [len(df), (756 - len(df))]
            labels = ['yes', 'no']
            fig, ax = mlt.subplots(figsize=(10, 6))
            ax = mlt.pie(slices, labels=labels,
                         autopct='%1.1f%%', startangle=90)
            plt.legend()
            st.pyplot(fig)

    if analysis == "General":
        st.title("General Analysis")
        col1, col2 = st.beta_columns(2)
        col3, col4 = st.beta_columns(2)
        with col1:
            st.write("**Total Matches Played each Season**")
            fig, ax = mlt.subplots(figsize=(10, 6))
            sns.countplot(x='season', data=matches,
                          palette=sns.color_palette('winter'))
            st.pyplot(fig)
            plt.xlabel("Season")
            plt.ylabel("Total Matches")
            st.text("")

            st.markdown(
                "Season 2011, 2012 and 2013 had the most most matches per season.")
            st.text("")
            st.text("")

        with col2:
            st.write("**Runs across all seasons**")

            batsmen = matches[['id', 'season']].merge(
                deliveries, left_on='id', right_on='match_id', how='left').drop('id', axis=1)
            season = batsmen.groupby(['season'])[
                'total_runs'].sum().reset_index()
            fig, ax = plt.subplots()
            season.set_index('season').plot(marker='o')
            plt.xlabel("Season")
            plt.ylabel("Total Runs")

            st.pyplot()

        with col3:
            # Man of the Match Analysis
            st.write("**Man of the Match Analysis**")
            mom_sel = st.slider("Number of players to show", 5, 50, 10)
            mom = matches.player_of_match.value_counts().reset_index()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax = sns.barplot(x="index", y="player_of_match",
                             data=mom[:mom_sel], palette=sns.color_palette('winter'))
            plt.xticks(rotation=90)
            plt.xlabel("Players")
            plt.ylabel("Man of the Match Awards")
            st.pyplot(fig)

            # Most Matches Won
            st.write("**Most matches won by a team**")
            matches_won = matches.winner.value_counts().reset_index()
            fig, ax = plt.subplots(figsize=(15, 15))
            ax = sns.barplot(x="index", y="winner",
                             data=matches_won, palette=sns.color_palette('winter'))
            plt.xticks(rotation=90)
            for p in ax.patches:
                ax.annotate(format(str(int(p.get_height()))), xy=(
                    p.get_x() + 0.25, p.get_height() + 1))
            plt.xlabel("Team")
            plt.ylabel("Number of Wins")
            st.pyplot(fig)

            # Top Grounds
        st.write("**Grounds with Most Matches**")
        grounds = matches['venue'].value_counts(
        ).rename_axis('venues').reset_index()
        grounds = grounds.rename({'venue': 'no_of_matches'}, axis=1)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax = sns.barplot(x="no_of_matches", y="venues",
                         data=grounds, palette=sns.color_palette("flare"))
        for p in ax.patches:
            width = p.get_width()
            plt.text((5 + p.get_width()), p.get_y() + 0.55 * p.get_height(),
                     '{:1.2f}'.format(width),
                     ha='center', va='center')
        plt.xlabel("Number of Matches")
        plt.ylabel("Venues")
        st.pyplot(fig)

        with col4:
            # Umpires
            st.write("**Umpires with Most Matches**")
            umpires = matches[['umpire1', 'umpire2']]
            ump1 = umpires.umpire1.value_counts()
            ump2 = umpires.umpire2.value_counts()
            ump1 = ump1.to_dict()
            ump2 = ump2.to_dict()
            ump = Counter(ump1) + Counter(ump2)
            ump = pd.DataFrame(ump.items(), columns=['umpire', 'matches'])
            ump = ump.sort_values(by=['matches'], ascending=0)

            fig, ax = plt.subplots(figsize=(20, 20))
            ump_select = st.slider("Number of Umpires", 5, 50, 10, key=11)
            ax = sns.barplot(x="umpire", y="matches",
                             data=ump[:ump_select], palette=sns.color_palette('winter', 10))
            plt.xticks(rotation=90)
            for p in ax.patches:
                ax.annotate(format(str(int(p.get_height()))), xy=(
                    p.get_x() + 0.25, p.get_height() + 1))
            plt.xlabel("Umpire")
            plt.ylabel("Matches")
            st.pyplot(fig)

            # Total Matches vs Wins

            st.write("**Total Matches vs Wins for Teams**")
            matches_played_byteams = pd.concat(
                [matches['team1'], matches['team2']])
            matches_played_byteams = matches_played_byteams.value_counts().reset_index()
            matches_played_byteams.columns = ['Team', 'Total Matches']
            matches_played_byteams['wins'] = matches['winner'].value_counts().reset_index()[
                'winner']
            matches_played_byteams.set_index('Team', inplace=True)

            trace1 = go.Bar(
                x=matches_played_byteams.index,
                y=matches_played_byteams['Total Matches'],
                name='Total Matches'
            )
            trace2 = go.Bar(
                x=matches_played_byteams.index,
                y=matches_played_byteams['wins'],
                name='Matches Won'
            )

            data = [trace1, trace2]
            layout = go.Layout(
                barmode='stack'
            )

            fig = go.Figure(data=data, layout=layout)
            st.plotly_chart(fig, filename='stacked-bar')

    if analysis == "Batsmen":
        st.title("Analysis of Batsmen")
        col1, col2 = st.beta_columns(2)

        orange_cap = matches[['id', 'season']]
        orange_cap = orange_cap.merge(
            deliveries, left_on='id', right_on='match_id', how='left')
        orange_cap = orange_cap.groupby(['season', 'batsman'])[
            'batsman_runs'].sum().reset_index()
        orange_cap = orange_cap.sort_values('batsman_runs', ascending=0)
        mrps = orange_cap

        # Most Runs Scored
        with col1:
            st.write("**Most runs scored by a batsman in all seasons**")
            no_of_batsman = st.slider(
                "Number of players to show", 5, 50, 15, key=1)
            runs_scored = deliveries.groupby(['batsman']).sum().sort_values([
                'batsman_runs'], ascending=False)
            runs_scored.reset_index(inplace=True)
            runs_scored = runs_scored[['batsman', 'batsman_runs']]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax = sns.barplot(x="batsman", y="batsman_runs",
                             data=runs_scored[:no_of_batsman])
            plt.xticks(rotation=90)
            plt.ylabel("Total Runs")
            for p in ax.patches:
                ax.annotate(str(int(p.get_height())),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points')
            st.pyplot(fig)
            st.text("")

            st.markdown("""Kohli has Scored the most runs followed by
                Raina, Sharma, Warner and Dhawan.""")
            st.text("")
            st.text("")

            # Top individual Scores
            st.write("**Top Individual Scores of all time**")
            st.text("")
            top_scores = deliveries.groupby(["match_id", "batsman", "batting_team"])[
                "batsman_runs"].sum().reset_index()
            new = top_scores.sort_values(['batsman_runs'], ascending=0)
            fig, ax = plt.subplots(figsize=(10, 6))

            ax = plt.bar("batsman", "batsman_runs",
                         data=new[:15], color='navy')

            for i in range(11):
                runs = new["batsman_runs"].iloc[i]
                plt.annotate(str(runs), (-0.15 + i, runs + 2))
            plt.xticks(rotation=90)
            plt.show()

            st.pyplot(fig)

        # Orange Cap
        with col2:
            # Most runs per season
            st.write("**Most runs per season**")

            index_for_mostruns = range(2008, 2020, 1)

            nbs = st.slider("Select Number of batsmen to show", 5, 50, 15)
            season = st.selectbox("Select a season", index_for_mostruns,
                                  len(index_for_mostruns) - 1)

            def ses(season):
                return mrps[mrps.season == season]

            df_runs = ses(season)[:nbs]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax = sns.barplot(x="batsman", y="batsman_runs",
                             data=df_runs, palette=sns.color_palette("rocket_r"))
            for p in ax.patches:
                ax.annotate(str(int(p.get_height())),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points')
            plt.xticks(rotation=90)
            st.pyplot(fig)

            st.text("")
            st.text("")
            st.write("**Orange Cap Winners per Season**")
            st.text("")
            orange_cap = orange_cap.drop_duplicates(
                subset=['season'], keep="first")
            orange_cap = orange_cap.sort_values(by='season')
            fig, ax = plt.subplots(figsize=(10, 6))
            ax = sns.barplot(x="season", y="batsman_runs", data=orange_cap,
                             color="darkorange")

            i = 0
            for p in ax.patches:
                ax.annotate(str(orange_cap.batsman.iloc[i]).split()[1]
                            + "\n" + str(int(p.get_height())),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points')

                i = i + 1

            st.pyplot(fig)

    if analysis == "Bowlers":
        st.title("Analysis of Bowlers")
        dismissal_kinds = ["bowled", "caught", "lbw",
                           "stumped", "caught and bowled", "hit wicket"]
        purple_cap = matches[['id', 'season']]
        purple_cap = purple_cap.merge(
            deliveries, left_on='id', right_on='match_id', how='left')

        purple_cap = purple_cap[deliveries.player_dismissed != 0]
        purple_cap = purple_cap[deliveries["dismissal_kind"].isin(
            dismissal_kinds)]

        for i in range(purple_cap.shape[0]):
            purple_cap.player_dismissed.iloc[i] = 1

        purple_cap = purple_cap.groupby(['season', 'bowler'])[
            'player_dismissed'].sum().reset_index()
        purple_cap = purple_cap.sort_values('player_dismissed', ascending=0)
        mwps = purple_cap

        col1, col2 = st.beta_columns(2)
        col3, col4 = st.beta_columns(2)
        with col1:
            # Most Wickets Ever

            st.write("Most wickets taken by a bowler in all seasons")
            no_of_bowlers = st.slider(
                "Select Number of Bowlers to Show", 5, 50, 15, key=11)
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            ct = deliveries[deliveries["dismissal_kind"].isin(dismissal_kinds)]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax = ct['bowler'].value_counts()[:no_of_bowlers].plot.bar(
                width=0.8, color=sns.color_palette('RdYlGn', 20))
            plt.ylabel("Innings")
            for p in ax.patches:
                ax.annotate(format(str(int(p.get_height()))),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points')
            st.pyplot(fig)

        with col2:
            # Most Wickets Per Season

            st.write("Most wickets per season")
            index_for_mostwickets = range(2008, 2020, 1)

            nws = st.slider("Select Number of Bowlers to Show",
                            5, 50, 15, key=2)
            season1 = st.selectbox("Select a season", index_for_mostwickets,
                                   len(index_for_mostwickets) - 1, key=2)

            def ses1(season):
                return mwps[mwps.season == season1]

            df_wickets = ses1(season1)[:nws]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax = sns.barplot(x="bowler", y="player_dismissed",
                             data=df_wickets, palette=sns.color_palette("rocket_r"))
            for p in ax.patches:
                ax.annotate(str(int(p.get_height())),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points')
            plt.xticks(rotation=90)
            st.pyplot(fig)

        # Purple Cap Holder
        with col3:
            st.write("Purple Cap Winners")

            purple_cap = purple_cap.drop_duplicates(
                subset=['season'], keep="first")
            purple_cap = purple_cap.sort_values(by='season')
            fig, ax = plt.subplots(figsize=(10, 6))
            ax = sns.barplot(x="season", y="player_dismissed", data=purple_cap,
                             color="purple")

            i = 0
            for p in ax.patches:
                ax.annotate(str(purple_cap.bowler.iloc[i]).split()[1],
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points')

                i = i + 1

            st.pyplot(fig)
            purple_cap = purple_cap.rename(
                {'player_dismissed': 'wickets'}, axis=1).reset_index()
            st.write(purple_cap)
        with col4:
            st.write("Types of Wickets")

            wickets = deliveries.dismissal_kind.value_counts()
            wickets = wickets.reset_index()
            index = wickets["index"].to_numpy()
            dismissal_types = wickets.dismissal_kind.to_numpy()
            pct = 100. * dismissal_types / dismissal_types.sum()

            labels = ['{0} - {1:1.2f} %'.format(i, j)
                      for i, j in zip(index, pct)]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax = plt.pie(dismissal_types, startangle=90)
            plt.legend(labels)
            plt.tight_layout()
            st.pyplot(fig)
            st.text("")
            st.text("")
            st.write(
                "** <-----------------------------------Purple Cap Winners------------------------------------ **")
