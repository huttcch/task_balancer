import pandas as pd
import numpy as np
import io
from flask import Flask, request, render_template, send_file

# --- ส่วนของ Genetic Algorithm ---

# 1. สร้าง Chromosome αρχικό (สุ่ม Label 1 ถึง N ให้กับแต่ละงาน)
def create_chromosome(num_tasks, n_people):
    return np.random.randint(1, n_people + 1, size=num_tasks)

# 2. คำนวณ Fitness Function (ความแปรปรวนของผลรวม Workload)
def calculate_fitness(chromosome, workloads, n_people):
    # สร้าง list ของผลรวม workload ของแต่ละคน (เริ่มต้นด้วย 0)
    sums = [0] * n_people
    for i, label in enumerate(chromosome):
        # label ที่ได้จาก chromosome จะเป็น 1, 2, ..., N
        # แต่ index ของ list คือ 0, 1, ..., N-1 จึงต้อง -1
        sums[label - 1] += workloads[i]
    
    # ค่า fitness คือความแปรปรวน ยิ่งต่ำยิ่งดี
    return np.var(sums)

# 3. Selection: คัดเลือก Chromosome ที่ดีที่สุด (Fitness ต่ำสุด)
def selection(population, fitness_scores, elite_size=50):
    sorted_indices = np.argsort(fitness_scores)
    # คัดเอา chromosome ที่ดีที่สุด (fitness ต่ำสุด) elite_size อันดับแรก
    return [population[i] for i in sorted_indices[:elite_size]]

# 4. Crossover: สร้าง Chromosome รุ่นใหม่โดยการผสมข้ามสายพันธุ์
def crossover(parent1, parent2):
    # เลือกจุดตัดแบบสุ่ม
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# 5. Mutation: การกลายพันธุ์เพื่อเพิ่มความหลากหลาย
def mutation(chromosome, n_people, mutation_rate=0.1):
    mutated_chromosome = np.copy(chromosome)
    for i in range(len(mutated_chromosome)):
        if np.random.rand() < mutation_rate:
            # Change Mutation: เปลี่ยน Label ของ Task แบบสุ่ม
            mutated_chromosome[i] = np.random.randint(1, n_people + 1)
        
        if np.random.rand() < mutation_rate / 2: # ให้ Swap เกิดน้อยกว่า
            # Swap Mutation: สลับตำแหน่ง Label ของ 2 Task
            swap_with = np.random.randint(0, len(mutated_chromosome))
            mutated_chromosome[i], mutated_chromosome[swap_with] = mutated_chromosome[swap_with], mutated_chromosome[i]
            
    return mutated_chromosome

# --- ฟังก์ชันหลักในการรัน Genetic Algorithm ---
def run_genetic_algorithm(workloads, n_people, generations=100, pool_size=1000):
    num_tasks = len(workloads)

    # 1. สร้างประชากรเริ่มต้น (Initial Population)
    population = [create_chromosome(num_tasks, n_people) for _ in range(pool_size)]

    for generation in range(generations):
        # 2. คำนวณ Fitness ของแต่ละ Chromosome
        fitness_scores = [calculate_fitness(chromo, workloads, n_people) for chromo in population]

        # 3. คัดเลือกกลุ่ม Elite (Chromosome ที่ดีที่สุด)
        elites = selection(population, fitness_scores, elite_size=50)

        # 4. สร้างประชากรรุ่นถัดไป
        next_generation = elites.copy() # นำกลุ่มที่ดีที่สุดไปสู่รุ่นถัดไปเลย

        # เติมประชากรที่เหลือด้วย Crossover และ Mutation
        while len(next_generation) < pool_size:
            # สุ่มพ่อแม่จากกลุ่ม Elites
            parent1, parent2 = elites[np.random.randint(0, len(elites))], elites[np.random.randint(0, len(elites))]
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            next_generation.append(mutation(child1, n_people))
            if len(next_generation) < pool_size:
                next_generation.append(mutation(child2, n_people))
        
        population = next_generation

    # ค้นหา Chromosome ที่ดีที่สุดในรุ่นสุดท้าย
    final_fitness_scores = [calculate_fitness(chromo, workloads, n_people) for chromo in population]
    best_chromosome_index = np.argmin(final_fitness_scores)
    best_chromosome = population[best_chromosome_index]
    
    return best_chromosome


# --- ส่วนของ Flask Web App ---
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 1. รับข้อมูลจากฟอร์ม
        file = request.files['file']
        n_people = int(request.form['people'])

        if not file:
            return "กรุณาอัปโหลดไฟล์ Excel", 400

        # 2. อ่านข้อมูลจาก Excel
        df = pd.read_excel(file)
        
        # ตรวจสอบคอลัมน์ที่จำเป็น
        if 'task' not in df.columns or 'workload' not in df.columns:
            return "ไฟล์ Excel ต้องมีคอลัมน์ 'task' และ 'workload'", 400

        workloads = df['workload'].tolist()
        
        if len(workloads) < n_people:
            return f"จำนวนงาน ({len(workloads)}) ต้องมากกว่าหรือเท่ากับจำนวนคน ({n_people})", 400

        # 3. รัน Genetic Algorithm เพื่อหาการแบ่งงานที่ดีที่สุด
        best_assignment = run_genetic_algorithm(workloads, n_people)
        
        # 4. เพิ่มคอลัมน์ label ใน DataFrame
        df['assigned_to_person'] = best_assignment

        # 5. สร้างไฟล์ Excel ในหน่วยความจำเพื่อส่งกลับ
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Task_Assignment')
        output.seek(0)
        
        return send_file(
            output,
            download_name='task_assignment_result.xlsx',
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    # สำหรับ method GET ให้แสดงหน้าเว็บปกติ
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)