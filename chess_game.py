"""
Advanced Chess Game with Complete Features
Requirements: pip install pygame
Run: python chess_game.py
"""

import pygame
import sys
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import copy

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8
SIDEBAR_WIDTH = WINDOW_WIDTH - BOARD_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BROWN = (240, 217, 181)
DARK_BROWN = (181, 136, 99)
HIGHLIGHT_COLOR = (255, 255, 0, 128)
VALID_MOVE_COLOR = (0, 255, 0, 64)
CHECK_COLOR = (255, 0, 0, 128)
LAST_MOVE_COLOR = (255, 255, 0, 64)
SIDEBAR_COLOR = (50, 50, 50)
TEXT_COLOR = (255, 255, 255)

class PieceType(Enum):
    PAWN = "pawn"
    KNIGHT = "knight"
    BISHOP = "bishop"
    ROOK = "rook"
    QUEEN = "queen"
    KING = "king"

class Color(Enum):
    WHITE = "white"
    BLACK = "black"

@dataclass
class Move:
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    piece: 'Piece'
    captured: Optional['Piece'] = None
    is_castling: bool = False
    is_en_passant: bool = False
    promotion_piece: Optional[PieceType] = None

class Piece:
    def __init__(self, piece_type: PieceType, color: Color, pos: Tuple[int, int]):
        self.type = piece_type
        self.color = color
        self.pos = pos
        self.has_moved = False
        
    def get_symbol(self):
        symbols = {
            PieceType.KING: "♔" if self.color == Color.WHITE else "♚",
            PieceType.QUEEN: "♕" if self.color == Color.WHITE else "♛",
            PieceType.ROOK: "♖" if self.color == Color.WHITE else "♜",
            PieceType.BISHOP: "♗" if self.color == Color.WHITE else "♝",
            PieceType.KNIGHT: "♘" if self.color == Color.WHITE else "♞",
            PieceType.PAWN: "♙" if self.color == Color.WHITE else "♟"
        }
        return symbols[self.type]
    
    def copy(self):
        new_piece = Piece(self.type, self.color, self.pos)
        new_piece.has_moved = self.has_moved
        return new_piece

class ChessBoard:
    def __init__(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.current_turn = Color.WHITE
        self.move_history = []
        self.captured_pieces = {Color.WHITE: [], Color.BLACK: []}
        self.en_passant_target = None
        self.last_move = None
        self.selected_piece = None
        self.valid_moves = []
        self.in_check = False
        self.checkmate = False
        self.stalemate = False
        self.setup_board()
        
    def setup_board(self):
        # Place pawns
        for i in range(8):
            self.board[1][i] = Piece(PieceType.PAWN, Color.BLACK, (1, i))
            self.board[6][i] = Piece(PieceType.PAWN, Color.WHITE, (6, i))
        
        # Place other pieces
        piece_order = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN,
                      PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK]
        
        for i, piece_type in enumerate(piece_order):
            self.board[0][i] = Piece(piece_type, Color.BLACK, (0, i))
            self.board[7][i] = Piece(piece_type, Color.WHITE, (7, i))
    
    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        if 0 <= row < 8 and 0 <= col < 8:
            return self.board[row][col]
        return None
    
    def is_valid_position(self, row: int, col: int) -> bool:
        return 0 <= row < 8 and 0 <= col < 8
    
    def get_valid_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        moves = []
        row, col = piece.pos
        
        if piece.type == PieceType.PAWN:
            moves = self._get_pawn_moves(piece)
        elif piece.type == PieceType.KNIGHT:
            moves = self._get_knight_moves(piece)
        elif piece.type == PieceType.BISHOP:
            moves = self._get_bishop_moves(piece)
        elif piece.type == PieceType.ROOK:
            moves = self._get_rook_moves(piece)
        elif piece.type == PieceType.QUEEN:
            moves = self._get_queen_moves(piece)
        elif piece.type == PieceType.KING:
            moves = self._get_king_moves(piece)
        
        # Filter out moves that would leave king in check
        valid_moves = []
        for move in moves:
            if self._is_move_safe(piece, move):
                valid_moves.append(move)
        
        return valid_moves
    
    def _get_pawn_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        moves = []
        row, col = piece.pos
        direction = -1 if piece.color == Color.WHITE else 1
        
        # Move forward one square
        new_row = row + direction
        if self.is_valid_position(new_row, col) and self.board[new_row][col] is None:
            moves.append((new_row, col))
            
            # Move forward two squares from starting position
            start_row = 6 if piece.color == Color.WHITE else 1
            if row == start_row and self.board[new_row + direction][col] is None:
                moves.append((new_row + direction, col))
        
        # Capture diagonally
        for dc in [-1, 1]:
            new_col = col + dc
            if self.is_valid_position(new_row, new_col):
                target = self.board[new_row][new_col]
                if target and target.color != piece.color:
                    moves.append((new_row, new_col))
        
        # En passant
        if self.en_passant_target:
            en_row, en_col = self.en_passant_target
            if abs(en_col - col) == 1 and new_row == en_row:
                moves.append((en_row, en_col))
        
        return moves
    
    def _get_knight_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        moves = []
        row, col = piece.pos
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        
        for dr, dc in knight_moves:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_position(new_row, new_col):
                target = self.board[new_row][new_col]
                if not target or target.color != piece.color:
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_bishop_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        moves = []
        row, col = piece.pos
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for dr, dc in directions:
            for i in range(1, 8):
                new_row, new_col = row + dr * i, col + dc * i
                if not self.is_valid_position(new_row, new_col):
                    break
                
                target = self.board[new_row][new_col]
                if not target:
                    moves.append((new_row, new_col))
                else:
                    if target.color != piece.color:
                        moves.append((new_row, new_col))
                    break
        
        return moves
    
    def _get_rook_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        moves = []
        row, col = piece.pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            for i in range(1, 8):
                new_row, new_col = row + dr * i, col + dc * i
                if not self.is_valid_position(new_row, new_col):
                    break
                
                target = self.board[new_row][new_col]
                if not target:
                    moves.append((new_row, new_col))
                else:
                    if target.color != piece.color:
                        moves.append((new_row, new_col))
                    break
        
        return moves
    
    def _get_queen_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        return self._get_bishop_moves(piece) + self._get_rook_moves(piece)
    
    def _get_king_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        moves = []
        row, col = piece.pos
        
        # Normal king moves
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                new_row, new_col = row + dr, col + dc
                if self.is_valid_position(new_row, new_col):
                    target = self.board[new_row][new_col]
                    if not target or target.color != piece.color:
                        moves.append((new_row, new_col))
        
        # Castling
        if not piece.has_moved and not self.in_check:
            # King-side castling
            rook = self.board[row][7]
            if rook and rook.type == PieceType.ROOK and not rook.has_moved:
                if all(self.board[row][col] is None for col in range(5, 7)):
                    if not self._is_square_attacked((row, 5), piece.color):
                        moves.append((row, 6))
            
            # Queen-side castling
            rook = self.board[row][0]
            if rook and rook.type == PieceType.ROOK and not rook.has_moved:
                if all(self.board[row][col] is None for col in range(1, 4)):
                    if not self._is_square_attacked((row, 3), piece.color):
                        moves.append((row, 2))
        
        return moves
    
    def _is_move_safe(self, piece: Piece, move: Tuple[int, int]) -> bool:
        # Make a temporary move to check if it leaves king in check
        temp_board = copy.deepcopy(self)
        temp_board.make_move(piece.pos, move, skip_validation=True)
        
        # Find king position
        king_pos = None
        for row in range(8):
            for col in range(8):
                p = temp_board.board[row][col]
                if p and p.type == PieceType.KING and p.color == piece.color:
                    king_pos = (row, col)
                    break
        
        if king_pos:
            return not temp_board._is_square_attacked(king_pos, piece.color)
        return False
    
    def _is_square_attacked(self, pos: Tuple[int, int], defending_color: Color) -> bool:
        attacking_color = Color.BLACK if defending_color == Color.WHITE else Color.WHITE
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece and piece.color == attacking_color:
                    if self._can_piece_attack(piece, pos):
                        return True
        return False
    
    def _can_piece_attack(self, piece: Piece, target_pos: Tuple[int, int]) -> bool:
        row, col = piece.pos
        target_row, target_col = target_pos
        
        if piece.type == PieceType.PAWN:
            direction = -1 if piece.color == Color.WHITE else 1
            return target_row == row + direction and abs(target_col - col) == 1
        
        elif piece.type == PieceType.KNIGHT:
            return (abs(target_row - row), abs(target_col - col)) in [(1, 2), (2, 1)]
        
        elif piece.type in [PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN]:
            if piece.type == PieceType.BISHOP:
                if abs(target_row - row) != abs(target_col - col):
                    return False
            elif piece.type == PieceType.ROOK:
                if target_row != row and target_col != col:
                    return False
            else:  # Queen
                if target_row != row and target_col != col and abs(target_row - row) != abs(target_col - col):
                    return False
            
            # Check path is clear
            dr = 0 if target_row == row else (1 if target_row > row else -1)
            dc = 0 if target_col == col else (1 if target_col > col else -1)
            
            current_row, current_col = row + dr, col + dc
            while (current_row, current_col) != (target_row, target_col):
                if self.board[current_row][current_col] is not None:
                    return False
                current_row += dr
                current_col += dc
            return True
        
        elif piece.type == PieceType.KING:
            return abs(target_row - row) <= 1 and abs(target_col - col) <= 1
        
        return False
    
    def make_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                  promotion_piece: Optional[PieceType] = None, skip_validation: bool = False) -> bool:
        if not skip_validation:
            piece = self.board[from_pos[0]][from_pos[1]]
            if not piece or piece.color != self.current_turn:
                return False
            
            valid_moves = self.get_valid_moves(piece)
            if to_pos not in valid_moves:
                return False
        
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        piece = self.board[from_row][from_col]
        captured = self.board[to_row][to_col]
        
        # Handle en passant
        is_en_passant = False
        if piece.type == PieceType.PAWN and self.en_passant_target == (to_row, to_col):
            captured_row = from_row + (1 if piece.color == Color.WHITE else -1)
            captured = self.board[captured_row][to_col]
            self.board[captured_row][to_col] = None
            is_en_passant = True
        
        # Handle castling
        is_castling = False
        if piece.type == PieceType.KING and abs(to_col - from_col) == 2:
            is_castling = True
            if to_col == 6:  # King-side
                rook = self.board[from_row][7]
                self.board[from_row][5] = rook
                self.board[from_row][7] = None
                rook.pos = (from_row, 5)
                rook.has_moved = True
            else:  # Queen-side
                rook = self.board[from_row][0]
                self.board[from_row][3] = rook
                self.board[from_row][0] = None
                rook.pos = (from_row, 3)
                rook.has_moved = True
        
        # Update en passant target
        self.en_passant_target = None
        if piece.type == PieceType.PAWN and abs(to_row - from_row) == 2:
            self.en_passant_target = ((from_row + to_row) // 2, from_col)
        
        # Handle pawn promotion
        if piece.type == PieceType.PAWN and (to_row == 0 or to_row == 7):
            if promotion_piece:
                piece = Piece(promotion_piece, piece.color, (to_row, to_col))
                piece.has_moved = True
            else:
                piece = Piece(PieceType.QUEEN, piece.color, (to_row, to_col))
                piece.has_moved = True
        
        # Make the move
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = None
        piece.pos = (to_row, to_col)
        piece.has_moved = True
        
        # Record captured pieces
        if captured:
            self.captured_pieces[piece.color].append(captured)
        
        # Record move
        move = Move(from_pos, to_pos, piece, captured, is_castling, is_en_passant)
        self.move_history.append(move)
        self.last_move = move
        
        # Switch turns
        self.current_turn = Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
        
        # Check game state
        self._update_game_state()
        
        return True
    
    def _update_game_state(self):
        # Check if current player is in check
        self.in_check = False
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece and piece.type == PieceType.KING and piece.color == self.current_turn:
                    if self._is_square_attacked((row, col), self.current_turn):
                        self.in_check = True
                    break
        
        # Check for checkmate or stalemate
        has_valid_moves = False
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece and piece.color == self.current_turn:
                    if self.get_valid_moves(piece):
                        has_valid_moves = True
                        break
            if has_valid_moves:
                break
        
        if not has_valid_moves:
            if self.in_check:
                self.checkmate = True
            else:
                self.stalemate = True

class ChessGUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Advanced Chess Game")
        self.clock = pygame.time.Clock()
        self.board = ChessBoard()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.dragging = False
        self.drag_piece = None
        self.drag_pos = None
        self.promotion_dialog = False
        self.promotion_pos = None
        
    def draw_board(self):
        # Draw squares
        for row in range(8):
            for col in range(8):
                color = LIGHT_BROWN if (row + col) % 2 == 0 else DARK_BROWN
                rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.screen, color, rect)
        
        # Highlight last move
        if self.board.last_move:
            for pos in [self.board.last_move.from_pos, self.board.last_move.to_pos]:
                row, col = pos
                s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
                s.set_alpha(64)
                s.fill((255, 255, 0))
                self.screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))
        
        # Highlight selected piece
        if self.board.selected_piece:
            row, col = self.board.selected_piece.pos
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
            s.set_alpha(128)
            s.fill((255, 255, 0))
            self.screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))
        
        # Show valid moves
        for move in self.board.valid_moves:
            row, col = move
            center = (col * SQUARE_SIZE + SQUARE_SIZE // 2, 
                     row * SQUARE_SIZE + SQUARE_SIZE // 2)
            pygame.draw.circle(self.screen, (0, 200, 0), center, 10)
        
        # Highlight king in check
        if self.board.in_check:
            for row in range(8):
                for col in range(8):
                    piece = self.board.get_piece(row, col)
                    if piece and piece.type == PieceType.KING and piece.color == self.board.current_turn:
                        s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
                        s.set_alpha(128)
                        s.fill((255, 0, 0))
                        self.screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))
    
    def draw_pieces(self):
        for row in range(8):
            for col in range(8):
                piece = self.board.get_piece(row, col)
                if piece and not (self.dragging and piece.pos == self.drag_pos):
                    self.draw_piece(piece, col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                   row * SQUARE_SIZE + SQUARE_SIZE // 2)
    
    def draw_piece(self, piece: Piece, x: int, y: int):
        text = self.font_large.render(piece.get_symbol(), True, BLACK)
        text_rect = text.get_rect(center=(x, y))
        self.screen.blit(text, text_rect)
    
    def draw_sidebar(self):
        # Draw sidebar background
        sidebar_rect = pygame.Rect(BOARD_SIZE, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, SIDEBAR_COLOR, sidebar_rect)
        
        # Current turn
        y_offset = 20
        turn_text = f"Turn: {self.board.current_turn.value.capitalize()}"
        text = self.font_medium.render(turn_text, True, TEXT_COLOR)
        self.screen.blit(text, (BOARD_SIZE + 20, y_offset))
        
        # Game status
        y_offset += 50
        if self.board.checkmate:
            winner = "Black" if self.board.current_turn == Color.WHITE else "White"
            status_text = f"Checkmate! {winner} wins!"
            color = (255, 215, 0)  # Gold
        elif self.board.stalemate:
            status_text = "Stalemate!"
            color = (200, 200, 200)
        elif self.board.in_check:
            status_text = "Check!"
            color = (255, 100, 100)
        else:
            status_text = "Playing"
            color = (100, 255, 100)
        
        text = self.font_medium.render(status_text, True, color)
        self.screen.blit(text, (BOARD_SIZE + 20, y_offset))
        
        # Captured pieces
        y_offset += 60
        text = self.font_small.render("Captured:", True, TEXT_COLOR)
        self.screen.blit(text, (BOARD_SIZE + 20, y_offset))
        
        y_offset += 30
        for color in [Color.WHITE, Color.BLACK]:
            pieces_text = " ".join([p.get_symbol() for p in self.board.captured_pieces[color]])
            if pieces_text:
                text = self.font_medium.render(pieces_text, True, TEXT_COLOR)
                self.screen.blit(text, (BOARD_SIZE + 20, y_offset))
                y_offset += 40
        
        # Move history
        y_offset += 20
        text = self.font_small.render("Move History:", True, TEXT_COLOR)
        self.screen.blit(text, (BOARD_SIZE + 20, y_offset))
        
        y_offset += 30
        moves_to_show = self.board.move_history[-10:]  # Show last 10 moves
        for i, move in enumerate(moves_to_show):
            move_text = f"{i+1}. {self._format_move(move)}"
            text = self.font_small.render(move_text, True, TEXT_COLOR)
            self.screen.blit(text, (BOARD_SIZE + 20, y_offset))
            y_offset += 25
            if y_offset > WINDOW_HEIGHT - 50:
                break
        
        # Controls
        y_offset = WINDOW_HEIGHT - 100
        controls = [
            "R - Reset Game",
            "U - Undo Move",
            "ESC - Quit"
        ]
        for control in controls:
            text = self.font_small.render(control, True, TEXT_COLOR)
            self.screen.blit(text, (BOARD_SIZE + 20, y_offset))
            y_offset += 25
    
    def _format_move(self, move: Move) -> str:
        from_col = chr(ord('a') + move.from_pos[1])
        from_row = str(8 - move.from_pos[0])
        to_col = chr(ord('a') + move.to_pos[1])
        to_row = str(8 - move.to_pos[0])
        
        piece_symbol = ""
        if move.piece.type != PieceType.PAWN:
            piece_symbol = move.piece.type.value[0].upper()
        
        capture = "x" if move.captured else ""
        castle = "O-O" if move.is_castling and move.to_pos[1] > move.from_pos[1] else ""
        castle = "O-O-O" if move.is_castling and move.to_pos[1] < move.from_pos[1] else castle
        
        if castle:
            return castle
        else:
            return f"{piece_symbol}{from_col}{from_row}{capture}{to_col}{to_row}"
    
    def draw_promotion_dialog(self):
        # Draw semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Draw dialog box
        dialog_width = 400
        dialog_height = 200
        dialog_x = (WINDOW_WIDTH - dialog_width) // 2
        dialog_y = (WINDOW_HEIGHT - dialog_height) // 2
        
        pygame.draw.rect(self.screen, WHITE, (dialog_x, dialog_y, dialog_width, dialog_height))
        pygame.draw.rect(self.screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 3)
        
        # Draw title
        title = self.font_medium.render("Choose Promotion Piece", True, BLACK)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, dialog_y + 40))
        self.screen.blit(title, title_rect)
        
        # Draw piece options
        pieces = [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT]
        color = self.board.current_turn
        symbols = {
            PieceType.QUEEN: "♕" if color == Color.WHITE else "♛",
            PieceType.ROOK: "♖" if color == Color.WHITE else "♜",
            PieceType.BISHOP: "♗" if color == Color.WHITE else "♝",
            PieceType.KNIGHT: "♘" if color == Color.WHITE else "♞"
        }
        
        for i, piece_type in enumerate(pieces):
            x = dialog_x + 50 + i * 80
            y = dialog_y + 100
            
            # Draw button
            button_rect = pygame.Rect(x - 30, y - 30, 60, 60)
            pygame.draw.rect(self.screen, LIGHT_BROWN, button_rect)
            pygame.draw.rect(self.screen, BLACK, button_rect, 2)
            
            # Draw piece symbol
            text = self.font_large.render(symbols[piece_type], True, BLACK)
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)
    
    def get_square_from_mouse(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        x, y = pos
        if x < BOARD_SIZE and y < BOARD_SIZE:
            col = x // SQUARE_SIZE
            row = y // SQUARE_SIZE
            return (row, col)
        return None
    
    def handle_click(self, pos: Tuple[int, int]):
        if self.promotion_dialog:
            # Handle promotion selection
            dialog_x = (WINDOW_WIDTH - 400) // 2
            dialog_y = (WINDOW_HEIGHT - 200) // 2
            
            pieces = [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT]
            for i, piece_type in enumerate(pieces):
                x = dialog_x + 50 + i * 80
                y = dialog_y + 100
                button_rect = pygame.Rect(x - 30, y - 30, 60, 60)
                
                if button_rect.collidepoint(pos):
                    if self.promotion_pos:
                        self.board.make_move(self.promotion_pos[0], self.promotion_pos[1], piece_type)
                    self.promotion_dialog = False
                    self.promotion_pos = None
                    return
            return
        
        square = self.get_square_from_mouse(pos)
        if not square:
            return
        
        row, col = square
        piece = self.board.get_piece(row, col)
        
        if self.board.selected_piece:
            # Try to move the selected piece
            if (row, col) in self.board.valid_moves:
                from_pos = self.board.selected_piece.pos
                
                # Check for pawn promotion
                if self.board.selected_piece.type == PieceType.PAWN:
                    if (row == 0 and self.board.selected_piece.color == Color.WHITE) or \
                       (row == 7 and self.board.selected_piece.color == Color.BLACK):
                        self.promotion_dialog = True
                        self.promotion_pos = (from_pos, (row, col))
                        self.board.selected_piece = None
                        self.board.valid_moves = []
                        return
                
                self.board.make_move(from_pos, (row, col))
                self.board.selected_piece = None
                self.board.valid_moves = []
            elif piece and piece.color == self.board.current_turn:
                # Select a new piece
                self.board.selected_piece = piece
                self.board.valid_moves = self.board.get_valid_moves(piece)
            else:
                # Deselect
                self.board.selected_piece = None
                self.board.valid_moves = []
        elif piece and piece.color == self.board.current_turn:
            # Select a piece
            self.board.selected_piece = piece
            self.board.valid_moves = self.board.get_valid_moves(piece)
    
    def undo_move(self):
        if len(self.board.move_history) > 0:
            # Simple undo by resetting the board and replaying all moves except the last one
            self.board = ChessBoard()
            moves_to_replay = self.board.move_history[:-1]
            for move in moves_to_replay:
                self.board.make_move(move.from_pos, move.to_pos, skip_validation=True)
    
    def reset_game(self):
        self.board = ChessBoard()
        self.promotion_dialog = False
        self.promotion_pos = None
    
    def run(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_u:
                        self.undo_move()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging and self.drag_piece:
                        self.drag_pos = event.pos
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and self.dragging:
                        self.dragging = False
                        square = self.get_square_from_mouse(event.pos)
                        if square and self.board.selected_piece:
                            if square in self.board.valid_moves:
                                from_pos = self.board.selected_piece.pos
                                self.board.make_move(from_pos, square)
                        self.board.selected_piece = None
                        self.board.valid_moves = []
                        self.drag_piece = None
            
            # Draw everything
            self.screen.fill(WHITE)
            self.draw_board()
            self.draw_pieces()
            
            # Draw dragging piece
            if self.dragging and self.drag_piece:
                x, y = pygame.mouse.get_pos()
                self.draw_piece(self.drag_piece, x, y)
            
            self.draw_sidebar()
            
            if self.promotion_dialog:
                self.draw_promotion_dialog()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

def main():
    game = ChessGUI()
    game.run()

if __name__ == "__main__":
    main()
