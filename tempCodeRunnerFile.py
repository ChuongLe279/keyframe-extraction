            line.set_data(x_col, y_col)
            ax.set_xlim(x_col[0], x_col[-1])
            fig.canvas.draw_idle()
            fig.canvas.flush_events()