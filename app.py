import pandas as pd
import numpy as np
import random
from flask import Flask, render_template, request, jsonify, send_file
import io

# --- Genetic Algorithm Configuration ---
POPULATION_SIZE = 1000  # จำนวน Chromosome ใน Pool
SELECTION_RATE = 50     # คัด Chromosome ที่ดีที่สุด 50 อันดับ
MUTATION_RATE = 0.1     # โอกาสที่จะเกิด Mutation
NUM_GENERATIONS = 100   # จำนวน Generation ที่จะรัน

app = Flask(__name__)

# --- Genetic Algorithm Functions ---

def create_chromosome(num_tasks, num_groups):
    """สร้าง Chromosome (sequence ของ label) แบบสุ่ม"""
    return [random.randint(1, num_groups) for _ in range(num_tasks)]

def calculate_fitness(chromosome, workloads, num_groups):
    """คำนวณ Fitness function (variance) - ค่ายิ่งต่ำยิ่งดี"""
    group_sums = [0] * num_groups
    for i, group_label in enumerate(chromosome):
        group_sums[group_label - 1] += workloads[i]
    return np.var(group_sums)

def selection(population_with_fitness):
    """คัดเลือก Chromosome ที่ดีที่สุด (variance ต่ำสุด)"""
    sorted_population = sorted(population_with_fitness, key=lambda x: x[1])
    return [item[0] for item in sorted_population[:SELECTION_RATE]]

def crossover(parent1, parent2):
    """สร้างลูกจากการผสมข้ามของพ่อแม่"""
    if len(parent1) != len(parent2):
        return parent1, parent2 # Should not happen
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def change_mutation(chromosome, num_groups):
    """เปลี่ยน label ของ task หนึ่งแบบสุ่ม"""
    index = random.randint(0, len(chromosome) - 1)
    chromosome[index] = random.randint(1, num_groups)
    return chromosome

def swap_mutation(chromosome):
    """สลับ label ระหว่าง 2 task"""
    idx1, idx2 = random.sample(range(len(chromosome)), 2)
    chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome
    
# --- Flask Routes ---

@app.route('/')
def index():
    """แสดงหน้าเว็บหลัก"""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_file():
    """รับไฟล์ Excel, จำนวนคน และรัน GA"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        num_groups = int(request.form.get('num_people'))
        if num_groups <= 0:
            raise ValueError("Number of people must be positive")
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid number of people'}), 400

    try:
        df = pd.read_excel(file)
        if 'task' not in df.columns or 'workload' not in df.columns:
            return jsonify({'error': 'Excel file must have "task" and "workload" columns'}), 400
        
        workloads = df['workload'].tolist()
        num_tasks = len(workloads)

        if num_tasks < num_groups:
             return jsonify({'error': 'Number of tasks cannot be less than the number of people'}), 400

        # 1. Initialization - สร้างประชากรเริ่มต้น
        population = [create_chromosome(num_tasks, num_groups) for _ in range(POPULATION_SIZE)]

        # --- GA Loop ---
        for generation in range(NUM_GENERATIONS):
            # 2. Fitness Calculation
            population_with_fitness = [(chromo, calculate_fitness(chromo, workloads, num_groups)) for chromo in population]

            # 3. Selection
            selected_parents = selection(population_with_fitness)
            
            # 4. Create Next Generation
            next_population = selected_parents[:] # นำ elite (ที่ดีที่สุด) ไปยัง generation ถัดไปเลย

            while len(next_population) < POPULATION_SIZE:
                p1, p2 = random.sample(selected_parents, 2)
                
                # Crossover
                c1, c2 = crossover(p1, p2)
                
                # Mutation
                if random.random() < MUTATION_RATE:
                    c1 = change_mutation(c1, num_groups)
                if random.random() < MUTATION_RATE:
                    c2 = swap_mutation(c2)
                    
                next_population.extend([c1, c2])
            
            population = next_population[:POPULATION_SIZE]

        # --- สิ้นสุด GA ---
        # หา Chromosome ที่ดีที่สุด
        final_fitness = [(chromo, calculate_fitness(chromo, workloads, num_groups)) for chromo in population]
        best_chromosome, best_fitness = min(final_fitness, key=lambda x: x[1])

        # สร้างผลลัพธ์
        df['label'] = best_chromosome
        
        # คำนวณผลรวมของแต่ละกลุ่มเพื่อแสดงผล
        group_totals = df.groupby('label')['workload'].sum().to_dict()

        # เตรียมข้อมูลสำหรับส่งกลับไปหน้าเว็บ
        global result_df 
        result_df = df.copy() # เก็บ dataframe ไว้สำหรับดาวน์โหลด
        
        return jsonify({
            'table_html': df.to_html(classes='table table-striped', index=False),
            'group_totals': group_totals,
            'variance': best_fitness
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download')
def download_file():
    """ส่งไฟล์ผลลัพธ์ให้ User ดาวน์โหลด"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        result_df.to_excel(writer, index=False, sheet_name='Task_Allocation')
    
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='task_allocation_result.xlsx'
    )