



def get_counts_df(date, hour, two_min_slot):
    #date is like '2018-08-13'
    end_two_min = two_min_slot + 5
    sql = """
    select
        driver_id,
        broadcast_pings,
        geohash
    from
        data_analytics.ag_heatmap_quality_base
    where
        date_local = timestamp '{date}'
        and hour_local = {hour}
        and physical_vehicle_type = 'CAR'
        and two_min_slot_local = {two_min_slot}
        and state = 'IDLE'
        and city_id = 6
    """.format(date = date, hour=hour, two_min_slot=two_min_slot, end_two_min=end_two_min)

    df_pa = presto.fetch_df(sql)

    df_pa['has_job'] = np.where(df_pa['broadcast_pings']>0,1,0)
    f = {'has_job':['sum'], 'driver_id':['count']}
    df_pa = df_pa.groupby('geohash', as_index=False).agg(f)
    df_pa.columns = ['geohash','has_job','total_idle']
    df_pa = df_pa[df_pa['total_idle'] >= 3]
    df_pa['prob'] = df_pa['has_job']/df_pa['total_idle']
    
    return df_pa

def get_heatmap_df(day,run_time):
    #run time is like run_time = '2018-08-13 00:30:00'
    #day is like 13
    sql = """
    select 
        supply_pool,
        geohash,
        smoothed_score,
        case when smoothed_score <= 0.35 then 0 
             when smoothed_score >= 0.6 then 1
             else (smoothed_score - 0.35) / (0.6 - 0.35) 
        end as norm_score
    from data_science.xx_heatmap_smooth_prob 
    where 
        day= '{day}' and
        time_bucket = '{run_time}' and
        supply_pool = '[69,227,302]'
    """.format(run_time=run_time, day=day)

    df_hm = presto.fetch_df(sql)   
    
    df_hm = df_hm.groupby(['geohash'],as_index=False)['norm_score'].mean()
    
    return df_hm


def get_dfs():
    df_pa_1 = get_counts_df('2018-08-13', 8, 15)
    df_hm_1 = get_heatmap_df(13,'2018-08-13 00:30:00')
    
    df_pa_2 = get_counts_df('2018-08-13', 13, 15)
    df_hm_2 = get_heatmap_df(13,'2018-08-13 05:30:00')
    
    df_pa_3 = get_counts_df('2018-08-13', 18, 15)
    df_hm_3 = get_heatmap_df(13,'2018-08-13 10:30:00')
    
    return (df_pa_1, df_pa_2, df_pa_3, df_hm_1, df_hm_2, df_hm_3)

def choose_best_cutoffs(spacing, cols, df_pa_1, df_pa_2, df_pa_3, df_hm_1, df_hm_2, df_hm_3):
    cand_cutoffs = gen_cand_cutoffs(spacing, cols)
    best_corr = 0
    best_cutoff_idx = 0
    
    for i in range(len(cand_cutoffs)):
        df_hm_1['heatscore'] = df_hm_1['norm_score'].apply(lambda x: heat_score_transform(x,cand_cutoffs[i]))
        df_hm_2['heatscore'] = df_hm_2['norm_score'].apply(lambda x: heat_score_transform(x,cand_cutoffs[i]))
        df_hm_3['heatscore'] = df_hm_3['norm_score'].apply(lambda x: heat_score_transform(x,cand_cutoffs[i]))
    
        df_merged_1 = pd.merge(df_pa_1, df_hm_1, on="geohash", how="inner")
        df_merged_2 = pd.merge(df_pa_2, df_hm_2, on="geohash", how="inner")
        df_merged_3 = pd.merge(df_pa_3, df_hm_3, on="geohash", how="inner")
        
        corr_1 = df_merged_1['prob'].corr(df_merged_1['heatscore'])
        corr_2 = df_merged_2['prob'].corr(df_merged_2['heatscore'])
        corr_3 = df_merged_3['prob'].corr(df_merged_3['heatscore'])
        
        corr = corr_1 + corr_2 + corr_3
        
        if corr > best_corr:
            best_corr = corr
            best_cutoff_idx = i
    
    df_hm_1['heatscore'] = df_hm_1['norm_score'].apply(lambda x: heat_score_transform(x,cand_cutoffs[best_cutoff_idx]))
    df_hm_2['heatscore'] = df_hm_2['norm_score'].apply(lambda x: heat_score_transform(x,cand_cutoffs[best_cutoff_idx]))
    df_hm_3['heatscore'] = df_hm_3['norm_score'].apply(lambda x: heat_score_transform(x,cand_cutoffs[best_cutoff_idx]))

    df_merged_1 = pd.merge(df_pa_1, df_hm_1, on="geohash", how="inner")
    df_merged_2 = pd.merge(df_pa_2, df_hm_2, on="geohash", how="inner")
    df_merged_3 = pd.merge(df_pa_3, df_hm_3, on="geohash", how="inner")    
    
    jitter(df_merged_1['prob'], df_merged_1['heatscore'])
    jitter(df_merged_2['prob'], df_merged_2['heatscore'])
    jitter(df_merged_3['prob'], df_merged_3['heatscore'])
    
    return best_corr, cand_cutoffs[best_cutoff_idx]