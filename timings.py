

def time_intervals(time_scale):
    SEC_IN_MIN = 60
    SEC_IN_HOUR = 3600
    SEC_IN_DAY = 86400
    SEC_IN_WEEK = 86400 * 7
    SEC_IN_MONTH = 86400 * 30
    SEC_IN_YEAR = 86400 * 365
    SEC_IN_BIYEARLY = 86400 * 365 * 2
    
    
    
    if time_scale == "minutely":
        interval_size = SEC_IN_MIN
    elif time_scale == "hourly":
        interval_size = SEC_IN_HOUR
    elif time_scale == "2hourly":
        interval_size = 2*SEC_IN_HOUR
    elif time_scale == "4hourly":
        interval_size = 4*SEC_IN_HOUR
    elif time_scale == "6hourly":
        interval_size = 6*SEC_IN_HOUR
    elif time_scale == "12hourly":
        interval_size = 12*SEC_IN_HOUR
    elif time_scale == "daily":
        interval_size = SEC_IN_DAY
    elif time_scale == "2daily":
        interval_size = 2*SEC_IN_DAY
    elif time_scale == "4daily":
        interval_size = 4*SEC_IN_DAY
    elif time_scale == "weekly":
        interval_size = SEC_IN_WEEK
    elif time_scale == "monthly":
        interval_size = SEC_IN_MONTH
    elif time_scale == "yearly":
        interval_size = SEC_IN_YEAR
    elif time_scale == "biyearly":
        interval_size = SEC_IN_BIYEARLY
    return interval_size
