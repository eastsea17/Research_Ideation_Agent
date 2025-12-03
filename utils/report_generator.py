import os

def generate_html_report(evaluated_topics, filename="report.html"):
    """
    Generate an HTML report for the evaluated research topics.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Research Topic Brainstorming Report</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f4f4f9; }
            h1 { text-align: center; color: #2c3e50; margin-bottom: 40px; }
            .topic-card { background: #fff; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 30px; padding: 25px; border-left: 5px solid #3498db; }
            .topic-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 15px; margin-bottom: 15px; }
            .topic-title { font-size: 1.5em; color: #2c3e50; margin: 0; }
            .topic-score { background: #3498db; color: #fff; padding: 5px 15px; border-radius: 20px; font-weight: bold; }
            .section-title { font-weight: bold; color: #7f8c8d; margin-top: 15px; text-transform: uppercase; font-size: 0.9em; }
            .content { margin-top: 5px; }
            .toc-list { padding-left: 20px; }
            .evaluation-box { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 5px; padding: 15px; margin-top: 20px; }
            .score-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 10px; }
            .score-item { text-align: center; background: #fff; padding: 10px; border-radius: 4px; border: 1px solid #dee2e6; }
            .score-value { font-size: 1.2em; font-weight: bold; color: #2c3e50; }
            .score-label { font-size: 0.8em; color: #6c757d; }
            .paper-item { margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; border-left: 3px solid #3498db; }
            .paper-title { font-weight: bold; color: #2c3e50; margin-bottom: 5px; }
            .paper-title a { color: #3498db; text-decoration: none; }
            .paper-title a:hover { text-decoration: underline; }
            .paper-meta { font-size: 0.9em; color: #6c757d; margin-top: 3px; }
            .paper-meta-label { font-weight: bold; color: #7f8c8d; }
        </style>
    </head>
    <body>
        <h1>Research Topic Brainstorming Report</h1>
    """
    
    for i, item in enumerate(evaluated_topics):
        topic = item.topic
        eval_result = item.evaluation
        
        html_content += f"""
        <div class="topic-card">
            <div class="topic-header">
                <h2 class="topic-title">#{i+1} {topic.title}</h2>
                <div class="topic-score">Score: {eval_result.total_score}/15</div>
            </div>
            
            <div class="section-title">Background</div>
            <div class="content">{topic.background}</div>
            
            <div class="section-title">Necessity</div>
            <div class="content">{topic.necessity}</div>
            
            <div class="section-title">Table of Contents</div>
            <ul class="toc-list">
                {''.join([f'<li>{step}</li>' for step in topic.table_of_contents])}
            </ul>
            
            <div class="section-title">Expected Effects</div>
            <div class="content">{topic.expected_effects}</div>
            
            
            <div class="section-title">Related Papers</div>
            <div>
                {''.join([f'''
                <div class="paper-item">
                    <div class="paper-title">
                        {"<a href='" + paper["url"] + "' target='_blank'>" + paper["title"] + "</a>" if paper.get("url") else paper["title"]}
                    </div>
                    {f'<div class="paper-meta"><span class="paper-meta-label">Authors:</span> {", ".join(paper["authors"][:3]) + (" et al." if len(paper["authors"]) > 3 else "")}</div>' if paper.get("authors") else ''}
                    {f'<div class="paper-meta"><span class="paper-meta-label">Year:</span> {paper["year"]}</div>' if paper.get("year") else ''}
                    {f'<div class="paper-meta"><span class="paper-meta-label">Institutions:</span> {", ".join(paper["institutions"][:3]) + (" et al." if len(paper["institutions"]) > 3 else "")}</div>' if paper.get("institutions") else ''}
                </div>
                ''' for paper in topic.related_papers])}
            </div>
            
            <div class="evaluation-box">
                <div class="section-title" style="margin-top: 0;">Evaluation</div>
                <div class="score-grid">
                    <div class="score-item">
                        <div class="score-value">{eval_result.originality_score}</div>
                        <div class="score-label">Originality</div>
                    </div>
                    <div class="score-item">
                        <div class="score-value">{eval_result.feasibility_score}</div>
                        <div class="score-label">Feasibility</div>
                    </div>
                    <div class="score-item">
                        <div class="score-value">{eval_result.impact_score}</div>
                        <div class="score-label">Impact</div>
                    </div>
                </div>
                <p><strong>Reasoning:</strong> {eval_result.reasoning}</p>
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(filename, "w") as f:
        f.write(html_content)
    
    print(f"Report generated: {os.path.abspath(filename)}")
