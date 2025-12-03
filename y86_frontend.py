#!/usr/bin/env python3
"""y86_frontend.py（修正版）"""
from flask import Flask, request, jsonify, render_template_string
from y86_simulator_opt import run_with_trace

app = Flask(__name__)

# 修正JavaScript大括号冲突：用`{ }`替代`{{ }}`，或用模板转义
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Y86-64 Simulator</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f0f0f0; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .control-panel { margin-bottom: 20px; padding: 15px; background: #f8f8f8; border-radius: 6px; }
        .form-group { margin-bottom: 10px; }
        button { background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        button:hover { background: #45a049; }
        .section { margin-bottom: 15px; padding: 10px; background: #f8f8f8; border-radius: 6px; }
        .section h3 { margin-top: 0; color: #333; }
        .stat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin-top: 10px; }
        .stat-item { padding: 8px; background: #e9e9e9; border-radius: 4px; }
        pre { background: #f0f0f0; padding: 10px; border-radius: 4px; overflow-x: auto; max-height: 300px; overflow-y: auto; }
        .trace-nav { margin: 10px 0; }
        .trace-nav button { margin-right: 5px; background: #008CBA; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Y86-64 模拟器</h1>
        
        <div class="control-panel">
            <div class="form-group">
                <label>选择.yo文件:</label>
                <input type="file" id="yoFile" accept=".yo">
            </div>
            <div class="form-group">
                <label>
                    <input type="checkbox" id="useCache"> 启用Cache
                </label>
            </div>
            <button onclick="runSimulation()">运行模拟器</button>
        </div>

        <div class="results">
            <div class="section">
                <h3>基本信息</h3>
                <div class="stat-grid">
                    <div class="stat-item"><strong>Cache启用:</strong> <span id="cacheEnabled">否</span></div>
                    <div class="stat-item"><strong>内存类型:</strong> <span id="memoryType">N/A</span></div>
                    <div class="stat-item"><strong>执行步数:</strong> <span id="stepCount">0</span></div>
                    <div class="stat-item"><strong>最终状态:</strong> <span id="finalState">N/A</span></div>
                </div>
            </div>

            <div class="section" id="cacheSection" style="display: none;">
                <h3>Cache性能</h3>
                <div class="stat-grid">
                    <div class="stat-item"><strong>命中数:</strong> <span id="cacheHits">0</span></div>
                    <div class="stat-item"><strong>缺失数:</strong> <span id="cacheMisses">0</span></div>
                    <div class="stat-item"><strong>命中率:</strong> <span id="cacheHitRate">0%</span></div>
                    <div class="stat-item"><strong>性能评级:</strong> <span id="cachePerf">N/A</span></div>
                </div>
            </div>

            <div class="section">
                <h3>CPU状态 (最后一步)</h3>
                <div class="trace-nav">
                    <button onclick="showTraceStep(0)">第一步</button>
                    <button onclick="showTraceStep(-1)">上一步</button>
                    <button onclick="showTraceStep(1)">下一步</button>
                    <button onclick="showTraceStep('last')">最后一步</button>
                    <span>当前步骤: <span id="currentStep">0</span>/<span id="totalSteps">0</span></span>
                </div>
                <pre id="cpuState">{}</pre>
            </div>
        </div>
    </div>

    <script>
        let traceData = [];
        let currentStep = 0;

        function runSimulation() {
            const fileInput = document.getElementById('yoFile');
            const useCache = document.getElementById('useCache').checked;
            const file = fileInput.files[0];
            
            if (!file) {
                alert('请选择.yo文件!');
                return;
            }

            const reader = new FileReader();
            reader.onload = function(e) {
                const yoText = e.target.result;
                fetch('/run', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        yo_text: yoText,
                        use_cache: useCache
                    }),
                })
                .then(response => response.json())
                .then(data => {
                    traceData = data.trace;
                    const summary = data.summary || {};
                    
                    // 更新基本信息
                    document.getElementById('cacheEnabled').textContent = useCache ? '是' : '否';
                    document.getElementById('memoryType').textContent = summary.memory_type || 'N/A';
                    document.getElementById('stepCount').textContent = traceData.length;
                    document.getElementById('finalState').textContent = traceData.length > 0 ? traceData[traceData.length-1].STAT_DESC : 'N/A';
                    document.getElementById('totalSteps').textContent = traceData.length;

                    // 更新Cache信息
                    const cacheSection = document.getElementById('cacheSection');
                    if (useCache && summary.cache) {
                        cacheSection.style.display = 'block';
                        document.getElementById('cacheHits').textContent = summary.cache.hits;
                        document.getElementById('cacheMisses').textContent = summary.cache.misses;
                        document.getElementById('cacheHitRate').textContent = summary.cache.hit_rate + '%';
                        document.getElementById('cachePerf').textContent = summary.cache.performance;
                    } else {
                        cacheSection.style.display = 'none';
                    }

                    // 显示第一步
                    currentStep = 0;
                    showTraceStep(0);
                });
            };
            reader.readAsText(file);
        }

        function showTraceStep(step) {
            if (traceData.length === 0) return;
            
            if (step === 'last') {
                currentStep = traceData.length - 1;
            }else if (step === 0) {
                currentStep = 0;
            }else {
                currentStep = Math.max(0, Math.min(traceData.length - 1, currentStep + step));
            }
            
            
            const state = traceData[currentStep];
            document.getElementById('currentStep').textContent = currentStep + 1;
            document.getElementById('cpuState').textContent = JSON.stringify(state, null, 2);
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/run', methods=['POST'])
def run_simulation():
    data = request.get_json()
    yo_text = data.get('yo_text', '')
    use_cache = data.get('use_cache', False)
    
    trace, summary = run_with_trace(yo_text, use_cache=use_cache)
    return jsonify({'trace': trace, 'summary': summary})

if __name__ == '__main__':
    print("启动Y86-64模拟器Web前端: http://127.0.0.1:5000")
    app.run(debug=True)