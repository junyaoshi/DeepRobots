/**
* Anatomy of a page walk 
**/


//Wrap the offsets with the appropriate locks 
down_read(&target_mm->mmap_sem);
	pgd = pgd_offset(target_mm, va_curr);
	if (pgd_none(*pgd) || unlikely(pgd_bad(*pgd)))
		goto out_sem;
	up_read(&target_mm->mmap_sem);

static int ppagefs_walk_pgd(struct mm_struct *mm, unsigned long addr, unsigned long end,
			    enum ppagefs_file_type type,
			    struct list_head *pfn_list)
{
	pgd_t *pgd;
	unsigned long next;
	int count = 0;

	//Get the ptr to the pgd_struct from mm and addr
	pgd = pgd_offset(mm, addr);
	do {
		//This is the VA boundary that marks the end of the corresponding pgd page
		next = pgd_addr_end(addr, end);
		if (pgd_none_or_clear_bad(pgd))
			continue;
		//We pass in next to the p4d end range 
		//since we only want the p4ds corresponding to that particular pgd page
		count += ppagefs_walk_p4d(pgd, addr, next, type, pfn_list);
		//pgd increments by one since it's the pointer to the next pgd page
		//addr goes to next because it goes to the next corresponding VA
	} while (pgd++, addr = next, addr != end);

	return count;
}

//Example of the p4d
static int ppagefs_walk_p4d(pgd_t *pgd, unsigned long addr, unsigned long end,
			    enum ppagefs_file_type type,
			    struct list_head *pfn_list)
{
	p4d_t *p4d;
	unsigned long next;
	unsigned long start;
	int count = 0;

	start = addr;
	p4d = p4d_offset(pgd, addr);

	do {
		next = p4d_addr_end(addr, end);
		if (p4d_none_or_clear_bad(p4d))
			continue;
		count += ppagefs_walk_pud(p4d, addr, next, type, pfn_list);
	} while (p4d++, addr = next, addr != end);

	return count;
}

/**
* .
* .
* .
*/


//PTE level is a little different...
static int ppagefs_walk_pte(pmd_t *pmd, unsigned long addr, unsigned long end,
			    enum ppagefs_file_type type,
			    struct list_head *pfn_list)
{
	pte_t *pte;
	int count = 0;

	pte = pte_offset_map(pmd, addr);

	//This is the last level, so we just increment by page_size (always granular in terms of pages)

	while (addr < end) {
		if (!pte_none(*pte)) {
			unsigned long pfn = pte_pfn(*pte);

			//Thing you want to do with pte/pfn goes here
			count += ppagefs_count_page(pfn, pfn_list, type);
		}
		addr += PAGE_SIZE;
		pte = pte_offset_map(pmd, addr);
	}

	return count;
}

static int ppagefs_walk_pgtbl(pid_t pid, enum ppagefs_file_type type)
{
	struct task_struct *p = ppagefs_find_task_by_pid(pid);
	struct mm_struct *mm;
	struct vm_area_struct *vma;
	int count = 0;
	LIST_HEAD(pfn_list);

	mm = p->mm;
	if (!mm)
		return count;
	//grab relevant mm semaphore 
	down_read(&mm->mmap_sem);
	//iterate through all the vms of the mm 
	for (vma = mm->mmap; vma; vma = vma->vm_next) {
		count += ppagefs_walk_pgd(mm, vma->vm_start, vma->vm_end, type,
					  &pfn_list);
	}

	count += ppagefs_count_free_page_list(&pfn_list, type);
	up_read(&mm->mmap_sem);
	return count;
}

//=============================================================

//This function directly counts the number of pages that are only mapped to once,
//otherwise it sticks it in the list and counts unique elements in the list afterwards
static int ppagefs_count_page(unsigned long pfn, struct list_head *pfn_list,
			      enum ppagefs_file_type type)
{
	int ret = 0;
	struct page *page = pfn_to_page(pfn);

	switch (type) {
	case ppagefs_total:
		//_mapcount tells us how many things are being mapped to the page
		if (atomic_read(&page->_mapcount) > 1)
			ppagefs_add_pfn(pfn, pfn_list);
		else
			ret = 1;
		break;
	case ppagefs_zero:
		if (atomic_read(&page->_mapcount) > 1)
			ppagefs_add_pfn(pfn, pfn_list);
		else
			//Zero page check
			ret = is_zero_page(pfn);
		break;
	default:
		break;
	}
	return ret;
}

//=============================================================

