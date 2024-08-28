from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from model.sensor_model import predict

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file:
        file_path = os.path.join('data', file.filename)
        file.save(file_path)
        return redirect(url_for('forecast', filename=file.filename))
    
    return redirect(url_for('index'))

@app.route('/forecast', methods=['GET'])
def forecast():
    filename = request.args.get('filename')
    if not filename:
        return redirect(url_for('index'))
    file_path = os.path.join('data', filename)
    test, forecast, test_mape = predict(file_path)
    # Prepare comparison DataFrame
    test_comparison = pd.DataFrame({
        'Actual Test Data': np.round(test['Qty Invoiced'].values),
        'Predicted Values': np.round(forecast)
    }, index=test.index)
    # Aggregate to monthly data
    test_monthly = test_comparison.resample('M').sum()
    # Calculate Mean Absolute Percentage Error (MAPE)
    test_mape_monthly = np.mean(np.abs((test_monthly['Actual Test Data'] - test_monthly['Predicted Values']) / test_monthly['Actual Test Data'])) * 100
    test_mape_monthly = round(test_mape_monthly, 3)
    # Create Plotly figure
    fig = go.Figure()
    # Add Actual data trace
    fig.add_trace(go.Scatter(
        x=test_monthly.index,
        y=test_monthly['Actual Test Data'],
        mode='lines+markers',
        name='Actual',
        line=dict(color='#2C3E50', width=2),
        marker=dict(size=8, color='#2C3E50', opacity=0.8, line=dict(width=2, color='#1A242F')),
        hovertemplate='Date: %{x}<br>Actual: %{y:.0f}<extra></extra>'
    ))
    # Add Predicted data trace
    fig.add_trace(go.Scatter(
        x=test_monthly.index,
        y=test_monthly['Predicted Values'],
        mode='lines+markers',
        name='Predicted',
        line=dict(color='#E67E22', width=2),
        marker=dict(size=12, color='#E67E22', symbol='circle', opacity=0.8, line=dict(width=2, color='#D35400')),
        hovertemplate='Date: %{x}<br>Predicted: %{y:.0f}<extra></extra>'
    ))
    # Add Confidence Interval
    confidence_interval = 0.1 * test_monthly['Predicted Values']
    fig.add_trace(go.Scatter(
        x=test_monthly.index,
        y=test_monthly['Predicted Values'] + confidence_interval,
        mode='lines',
        name='Upper CI',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(230, 126, 34, 0.2)',
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=test_monthly.index,
        y=test_monthly['Predicted Values'] - confidence_interval,
        mode='lines',
        name='Lower CI',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(230, 126, 34, 0.2)',
        hoverinfo='skip'
    ))
    # Update layout
    fig.update_layout(
        title='Monthly Forecast vs Actual',
        xaxis_title='Date',
        yaxis_title='Qty Invoiced',
        legend_title='Legend',
        template='plotly',
        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(245, 245, 245, 1)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.8)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.8)',
            tickformat=',',
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        autosize=True
    )
    # Save plot as HTML file
    monthly_plot_path = 'static/monthly_forecast_plot.html'
    pio.write_html(fig, file=monthly_plot_path, auto_open=False)
    # Prepare data for rendering
    zipped_data = zip(
        test_monthly.index.strftime('%Y-%m'),
        test_monthly['Actual Test Data'],
        test_monthly['Predicted Values']
    )

    return render_template(
        'forecast.html',
        zipped_data=zipped_data,
        monthly_plot_path=url_for('static', filename='monthly_forecast_plot.html'),
        test_mape_monthly=test_mape_monthly
    )

if __name__ == '__main__':
    app.run(debug=True)
