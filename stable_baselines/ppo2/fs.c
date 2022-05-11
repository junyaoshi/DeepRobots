//hw6 review

#include <linux/dcache.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/magic.h>
#include <linux/pagemap.h>
#include <linux/sched.h>
#include <linux/sched/task.h>
#include <linux/slab.h>
#include <linux/stat.h>
#include <linux/string.h>
#include <linux/time.h>
#include <linux/uaccess.h>
#include <uapi/linux/mount.h>
#include <linux/seq_file.h>
#include <linux/mm.h>
#include <linux/sched/mm.h>
#include <asm/pgtable.h>
#include <linux/hugetlb.h>
#include <linux/hashtable.h>
#include <linux/fs_context.h>
#include <linux/list_sort.h>
#include <linux/sched/signal.h>

/**
*Required structs (and by extension, functions)
**/

static const struct file_operations ppagefs_file_ops = {
	.read		= ppagefs_file_read,
	.open		= ppagefs_file_open,
	.release	= ppagefs_file_release,
	.fsync		= noop_fsync,
};

static const struct file_operations ppagefs_dir_ops = {
	.open		= ppagefs_dir_open,
	.release	= dcache_dir_close,
	.llseek		= dcache_dir_lseek,
	.read		= generic_read_dir,
	.iterate	= dcache_readdir,
	.fsync		= noop_fsync,
};

static const struct super_operations ppagefs_sops = {
	.statfs		= simple_statfs,
	.drop_inode	= generic_delete_inode,
};


static struct file_system_type ppagefs_type = {
	.name = "ppagefs",
	.init_fs_context = ppagefs_init_fs_context,
	.kill_sb = kill_litter_super,
};


//inode struct link
https://elixir.bootlin.com/linux/v5.4/source/include/linux/fs.h#L628

//dentry struct link
https://elixir.bootlin.com/linux/v5.4/source/include/linux/dcache.h#L89

//inode creation
struct inode *inode = new_inode(sb);

inode->i_ino = get_next_ino();
inode_init_owner(inode, dir, mode);
inode->i_mode = S_IFREG | PPAGEFS_FILE_MODE;
inode->i_atime = inode->i_mtime = inode->i_ctime = current_time(inode);
inode->i_op = &ppagefs_inode_ops;
inode->i_fop = &ppagefs_file_ops;


//Helpful functions
dget_parent(dentry)

static void clear_root_dirs(struct dentry *parent)
{
	struct dentry *temp, *dentry;
	pid_t pid;
	struct task_struct *tsk;

	spin_lock(&parent->d_lock);
	list_for_each_entry_safe(dentry, temp, &parent->d_subdirs, d_child) {
		if (name_to_pid(dentry->d_name.name, &pid))
			continue;
		rcu_read_lock();
		tsk = ppagefs_find_task_by_pid(pid);
		if (tsk)
			get_task_struct(tsk);
		rcu_read_unlock();
		if (!tsk) {
			list_del(&dentry->d_child);
			d_genocide(dentry);
		}
		if (tsk)
			put_task_struct(tsk);
	}
	spin_unlock(&parent->d_lock);
}
