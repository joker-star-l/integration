from retree.parser import parse

def test_sqlparse():
    sql = '''SELECT count(*) AS weeks 
FROM weekly_status w JOIN stores s ON w.store = s.store
JOIN features f ON w.store = f.store AND w.date = f.date 
WHERE predict('dt.onnx', data) > 30000
GROUP BY w.store, w.dept;'''
    parsed, ml_precidates = parse(sql, 'predict')
    ml_precidate = ml_precidates[0]
    assert str(ml_precidate.token) == "predict('dt.onnx', data) > 30000"
    assert ml_precidate.get_model_path_str() == 'dt.onnx'
    assert ml_precidate.get_comparison_operator_str() == '>'
    assert ml_precidate.get_comparison_value_number() == 30000
    func = ml_precidate.get_func()
    assert func(30000) == False
    assert func(30001) == True
    ml_precidate.rewrite()
    assert str(ml_precidate.token) == "predict('out_dt.onnx', data) > 0.5"
    assert ml_precidate.get_model_path_str() == 'out_dt.onnx'
    assert ml_precidate.get_comparison_operator_str() == '>'
    assert ml_precidate.get_comparison_value_number() == 0.5
    out_sql = '''SELECT count(*) AS weeks 
FROM weekly_status w JOIN stores s ON w.store = s.store
JOIN features f ON w.store = f.store AND w.date = f.date 
WHERE predict('out_dt.onnx', data) > 0.5
GROUP BY w.store, w.dept;'''
    assert str(parsed) == out_sql

test_sqlparse()
